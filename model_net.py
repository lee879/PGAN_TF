import tensorflow as tf
import numpy as np
import tensorflow
from tensorflow.python.keras import Model,Sequential
from tensorflow.python.keras.layers import Dense,Conv2D,Activation,Layer,Conv2DTranspose,Flatten,UpSampling2D,MaxPooling2D,AveragePooling2D
from tensorflow.python.keras.models import save_model
factors_G_Channels = [512,512,512,512,256,128,64,32,16] # 一共有9个模块
factors_D_Channels = [16,32,64,128,256,512,512,512,512] # 一共有9个模块

# class WSConv2D(Layer):
#     def __init__(self, in_channels, out_channels, k=3, s=1, gain=2):
#         super(WSConv2D, self).__init__()
#         self.conv = tf.keras.layers.Conv2D(out_channels,
#                                            kernel_size=k,
#                                            strides=s,
#                                            padding="same",
#                                            use_bias=False)
#         self.scale = (gain / (in_channels * k ** 2)) ** 0.5 #给一个放缩系数,可以通过输入来调整
#         self.bias = self.add_weight(shape=(out_channels), initializer='zeros',trainable=True)
#
#     def call(self, x):
#         x= self.conv(x * self.scale) + self.bias
#
#         return x
class minibatch_std(Layer):
    def __init__(self,name,in_channels=None):
        super(minibatch_std, self).__init__()
        self.conv = WSConv2D(512,512,name=name,k=3, s=1,)
        #self.l = Dense(512)
    def call(self, inputs, *args, **kwargs):
        x = self.conv(inputs)
        out = self.minibatch_stddev_layer(x)
        return out
    def minibatch_stddev_layer(self,x, group_size=4, num_new_features=1): # 将一个数据拼接到最后面通过广播的形式 ？,4,4,512 --> ? ,4,4,513 小批量标准差
        x = tf.transpose(x,perm=[0, 3, 1, 2]) # [NHWC] --> [NCHW]
        group_size = tf.minimum(group_size,tf.shape(x)[0])               # Minibatch must be divisible by (or smaller than) group_size.
        s = x.shape                                                     # [NCHW]  Input shape.
        y = tf.reshape(x, [group_size, -1, num_new_features, s[1] // num_new_features, s[2], s[3]]) # [GMncHW] Split minibatch into M groups of size G. Split channels into n channel groups c.
        y = tf.cast(y, tf.float32)                                      # [GMncHW] Cast to FP32.
        y -= tf.reduce_mean(y, axis=0, keepdims=True)                   # [GMncHW] Subtract mean over group.
        y = tf.reduce_mean(tf.square(y), axis=0)                        # [MncHW]  Calc variance over group.
        y = tf.sqrt(y + 1e-8)                                           # [MncHW]  Calc stddev over group.
        y = tf.reduce_mean(y, axis=[2, 3, 4], keepdims=True)            # [Mn111]  Take average over fmaps and pixels.
        y = tf.reduce_mean(y, axis=[2])                                 # [Mn11] Split channels into c channel groups
        y = tf.cast(y, x.dtype)                                         # [Mn11]  Cast back to original data type.
        y = tf.tile(y, [group_size, 1, s[2], s[3]])                     # [NnHW]  Replicate over group and pixels.
        out = tf.transpose(tf.concat([x, y], axis=1),perm=[0, 2, 3, 1]) # [NCHW] --> [NHWC]
        return out
class WSConv2D(Layer):
    def __init__(self, in_channels, out_channels, name=1,k=3, s=1, gain=2):
        super(WSConv2D, self).__init__()
        self.conv = tf.keras.layers.Conv2D(out_channels,
                                           kernel_size=k,
                                           strides=s,
                                           padding="same",
                                           use_bias=False,
                                           kernel_initializer='random_normal')
        self.scale = (gain / (in_channels * k ** 2)) ** 0.5 #给一个放缩系数,可以通过输入来调整
        self.bias = self.add_weight(shape=(out_channels), initializer='zeros',trainable=True,name=str(name))

    def call(self, x):
        x = self.conv(x * self.scale) + tf.reshape(self.bias, (1, 1, 1, -1))
        return x
class PixelNorm(Layer):
    def __init__(self):
        super(PixelNorm, self).__init__()
        self.epsdion = 1e-8
    def call(self, x, **kwargs):
        #out = inputs / tf.sqrt(tf.reduce_mean(inputs ** 2) + self.epsdion)
        out = x / tf.math.sqrt(tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + self.epsdion )
        return out

class ConvBlock(Layer):
    def __init__(self,filters,name):
        super(ConvBlock, self).__init__()
        self.conv1 = WSConv2D(in_channels=filters,out_channels=filters,name=name)
        self.conv2 = WSConv2D(in_channels=filters,out_channels=filters,name=name+1)
        self.act = tensorflow.keras.layers.Activation(tf.nn.leaky_relu)
        self.swish = Swish()
        #self.bn = tensorflow.keras.layers.BatchNormalization()
        self.pn1 = tensorflow.keras.layers.BatchNormalization()
        self.pn2 = tensorflow.keras.layers.BatchNormalization()
        #self.pn = PixelNorm()

    def call(self, inputs, **kwargs):
        x = self.act(self.conv1(inputs))
        #x = self.swish(self.conv1(inputs))
        x = self.pn1(x)
        x = self.act(self.conv2(x))
        #x = self.swish(self.conv2(inputs))
        x = self.pn2(x)
        return x

class initModel_G(Model):
    def __init__(self,out_channels,name):
        super(initModel_G, self).__init__()
        #self.pn = PixelNorm()
        self.l1 = Conv2DTranspose(out_channels,4,4,padding="same",activation=tf.nn.leaky_relu)
        self.l2 = WSConv2D(out_channels,out_channels,name=name,k = 3,s = 1)
        self.ac = Activation(tf.nn.leaky_relu)
        self.out = WSConv2D(in_channels=out_channels,out_channels=3,name=name+1,k = 3,s = 1)
        self.ac_out = Activation(tf.nn.tanh)
        self.bn = tf.keras.layers.BatchNormalization()
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()

    def call(self, inputs, alpha=None,training=None, mask=None):
        x = self.bn(inputs)
        x = self.l1(x)
        x = self.bn1(x)
        x = self.l2(x)
        x = self.ac(x)
        x = self.bn2(x)
        x = self.ac_out(self.out(x))
        return x


class G_Backbone(Model):
    def __init__(self,model,out_channels,name):
        super(G_Backbone, self).__init__()
        self.ini_mode = model
        self.up = UpSampling2D(2)
        self.convblock = ConvBlock(out_channels,name=name)
        self.out = WSConv2D(in_channels=3,out_channels=3,name=name+2,k=3,s=1)
        self.to_rgb = WSConv2D(in_channels=out_channels,out_channels=out_channels,name=name+5,k=1,s=1)
        self.ac = Activation(tf.nn.tanh)

    def call(self, inputs,alpha,training=None, mask=None):
        x = self.ini_mode(inputs,alpha)
        x = self.up(x)
        y = self.to_rgb(x)
        x = self.convblock(x)
        out = self.ac(self.out(alpha * y + (1-alpha) * x))
        return out
class initModel_D(Model):
    def __init__(self,out_channels,name):
        super(initModel_D, self).__init__()
        self.ms = minibatch_std(name)
        self.pn = PixelNorm()
        #self.bn = tensorflow.keras.layers.BatchNormalization()
        self.conv1 = WSConv2D(out_channels + 1,out_channels,name+1,3,1)
        self.ac = Activation(tf.nn.leaky_relu)
        self.conv2 = WSConv2D(out_channels, out_channels,name+2,4 ,1)
        #self.to_rgb = WSConv2D(in_channels=3, out_channels=3, name=name + 5, k=1, s=1)
        self.out=WSConv2D(out_channels, 1, name + 7, 3, 4)
        self.fd = Flatten()
        # self.fc = Dense(1)
    def call(self, inputs, alpha = None,training=None, mask=None):
        x = self.ms(inputs)
        x = self.conv1(x)
        x = self.ac(x)
        x = self.pn(x)
        x = self.conv2(x)
        x = self.ac(x)
        x = self.pn(x)
        x = self.out(x)
        x = self.fd(x)
        # x = self.fc(x)
        return x

#使用swish激活函数
class Swish(Layer):
    def __init__(self):
        super(Swish, self).__init__()

    def call(self, inputs):
        return inputs * tf.sigmoid(inputs)

class D_Backbone(Model):
    def __init__(self,model,out_channels,name):
        super(D_Backbone,self).__init__()
        self.to_rgb = WSConv2D(in_channels=out_channels, out_channels=3, name=name + 5, k=3, s=1)
        #self.to_rgb1 = ConvBlock(out_channels, name + 2)  # 为什么在鉴别器网路模块中有个3输出,应为初始模型输入的是channel3
        self.convblock = ConvBlock(out_channels,name)
        self.ini_mode = model
        #self.down = MaxPooling2D(2)
        self.down = AveragePooling2D(2)
        self.out = WSConv2D(3,3,name=name+9,k=3, s=1)
        self.ac = Activation(tf.nn.leaky_relu)
        #self.act = Activation(tf.nn.tanh)


    def call(self, inputs,alpha,training=None, mask=None):
        y = self.down(inputs)
        y = self.to_rgb(y)
        #x = self.to_rgb1(inputs)
        x = self.convblock(inputs)
        #x = self.out(x)
        x = self.down(x)
        x = self.out(x)
        x = tf.tanh(alpha * y + (1-alpha) * x)
        out = self.ini_mode(x,alpha)
        return out


if __name__== "__main__":
    name = int(1)
    model_g = initModel_G(factors_G_Channels[0],name=name) # 调用模型
    x = tf.random.normal([4,1,1,512])
    y = model_g(x)
    print(y.shape)

    name = name + 100
    model_d = initModel_D(factors_D_Channels[8], name=name)
    z = model_d(y)
    print(z.shape)

    name = name + 100
    for layer in model_g.layers:
        layer.trainable = False
    model_g = G_Backbone(model_g,factors_G_Channels[1],name=name) #
    y = model_g(x,alpha=0.5)
    print(y.shape)

    name = name + 100
    model_d = D_Backbone(model_d,factors_D_Channels[7],name=name) # 因为第一层输出的图片的是一个输入是一个4的
    z = model_d(y,alpha=0.5)
    print(z.shape)

    name = name + 100
    model_g = G_Backbone(model_g,factors_G_Channels[2],name=name) #
    y = model_g(x,alpha=0.5)
    print(y.shape)

    name = name + 100
    model_d = D_Backbone(model_d,factors_D_Channels[6],name=name) # 因为第一层输出的图片的是一个输入是一个4的
    z = model_d(y,alpha=0.5)
    print(z.shape)
    name = name + 100
    model_g = G_Backbone(model_g,factors_G_Channels[3],name=name) #
    y = model_g(x,alpha=0.5)
    print(y.shape)
    name = name + 100
    model_d = D_Backbone(model_d,factors_D_Channels[5],name=name) # 因为第一层输出的图片的是一个输入是一个4的
    z = model_d(y,alpha=0.5)
    print(z.shape)
    name = name + 100
    model_g = G_Backbone(model_g,factors_G_Channels[4],name=name) #
    y = model_g(x,alpha=0.5)
    print(y.shape)
    name = name + 100
    model_d = D_Backbone(model_d,factors_D_Channels[4],name=name) # 因为第一层输出的图片的是一个输入是一个4的
    z = model_d(y,alpha=0.5)
    print(z.shape)
    name = name + 100
    model_g = G_Backbone(model_g,factors_G_Channels[5],name=name) #
    y = model_g(x,alpha=0.5)
    print(y.shape)
    name = name + 100
    model_d = D_Backbone(model_d,factors_D_Channels[3],name=name) # 因为第一层输出的图片的是一个输入是一个4的
    z = model_d(y,alpha=0.5)
    print(z.shape)
    name = name + 100
    model_g = G_Backbone(model_g,factors_G_Channels[6],name=name) #
    y = model_g(x,alpha=0.5)
    print(y.shape)

    name = name + 100
    model_d = D_Backbone(model_d,factors_D_Channels[2],name=name) # 因为第一层输出的图片的是一个输入是一个4的
    z = model_d(y,alpha=0.5)
    print(z.shape)

    name = name + 100
    model_g = G_Backbone(model_g,factors_G_Channels[7],name=name) #
    y = model_g(x,alpha=0.5)
    print(y.shape)

    name = name + 100
    model_d = D_Backbone(model_d,factors_D_Channels[1],name=name) # 因为第一层输出的图片的是一个输入是一个4的
    z = model_d(y,alpha=0.5)
    print(z.shape)

    name = name + 100
    model_g_e = G_Backbone(model_g,factors_G_Channels[8],name=name) #
    y = model_g_e(x,alpha=0.5)
    print(y.shape)
    print("------------------------------------------------------------")
    for i, w in enumerate(model_g_e.weights):
        print(i, w.name)
    model_g_e.save_weights("1.h5")

    name = name + 100
    model_d_e = D_Backbone(model_d,factors_D_Channels[0],name=name) # 因为第一层输出的图片的是一个输入是一个4的
    z = model_d_e(y,alpha=0.5)
    print("------------------------------------------------------------")
    for i, w in enumerate(model_d_e.weights):
        print(i, w.name)
    print(z.shape)
    model_d_e.save_weights("1.h5")









