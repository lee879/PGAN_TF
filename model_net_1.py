import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers,Sequential

factors = [1,1,1,1,1/2,1/4,1/8,1/16,1/32]
#factors = [1,1,1,1,1/2,1/4]
class minibatch_std(keras.layers.Layer):
    def __init__(self):
        super(minibatch_std, self).__init__()
        #self.conv = WSConv2d(512,512,k=3, s=1,)
        #self.l = Dense(512)
    def call(self, inputs, *args, **kwargs):
       # x = self.conv(inputs)
        out = self.minibatch_stddev_layer(inputs)
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

# class WSConv2d(tf.keras.layers.Layer):
#     def __init__(self, in_channels, out_channels, name=1,k=3, s=1, gain=2):
#         super(WSConv2d, self).__init__()
#         self.conv = tf.keras.layers.Conv2D(out_channels,
#                                            kernel_size=k,
#                                            strides=s,
#                                            padding="same",
#                                            use_bias=True,
#                                            kernel_initializer='random_normal')
#         self.scale = (gain / (in_channels * k ** 2)) ** 0.5 #给一个放缩系数,可以通过输入来调整
#         #self.bias = self.add_weight(shape=(out_channels), initializer='zeros',trainable=True,name=str(name))
#         #self.bias = bia(out_channels)
#         #self.bias = self.add_variable(shape=out_channels,,trainable=True)
#
#     def call(self, x):
#         x = self.conv(x * self.scale)
#         return x

class WSConv2d(tf.keras.layers.Layer):
    def __init__(self, in_channels=None, out_channels=None,k=3, s=1, **kwargs):
        super(WSConv2d, self).__init__(**kwargs)
        self.filters = out_channels
        self.kernel_size = k
        self.conv2d = tf.keras.layers.Conv2D(out_channels, kernel_size=k,strides=s,padding="same")
        #self.gamma = self.add_weight(name='gamma', shape=(1,), trainable=True)

    def call(self, inputs):
        x = self.conv2d(inputs)
        scale = tf.reduce_mean(tf.abs(x), axis=[1, 2], keepdims=True)
        #scaled_x = x * self.gamma / scale
        scaled_x = x  / scale
        return scaled_x

#
# class WSConv2d(tf.keras.layers.Layer):
#     def __init__(self, in_channels=None, out_channels=None,k=3, s=1, **kwargs):
#         super(WSConv2d, self).__init__(**kwargs)
#         self.filters = out_channels
#         self.kernel_size = k
#         self.conv2d = tf.keras.layers.Conv2D(out_channels, kernel_size=k,strides=s,padding="same")
#         self.gamma = tf.Variable(initial_value=tf.ones((1,), dtype=tf.float32), trainable=True)
#
#     def call(self, inputs):
#         x = self.conv2d(inputs)
#         scale = tf.reduce_mean(tf.abs(x), axis=[1,2], keepdims=True)
#         scaled_x = x * self.gamma / scale
#         return scaled_x
class PixelNorm(layers.Layer):
    def __init__(self):
        super(PixelNorm, self).__init__()
        self.e = 1e-8

    def call(self, x):
        return x / tf.math.sqrt(tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + self.e) #


class ConvBlock(layers.Layer):
    def __init__(self,in_channels,out_channels,use_pn = True):
        super(ConvBlock, self).__init__()
        self.use_pn = use_pn
        self.conv1 = WSConv2d(in_channels,out_channels)
        self.conv2 = WSConv2d(out_channels, out_channels)
        self.leak = layers.LeakyReLU(0.2)
        self.pn=PixelNorm()

    def call(self,x):
        x = self.leak(self.conv1(x))
        x = self.pn(x) if self.use_pn else x
        x = self.leak(self.conv2(x))
        x = self.pn(x) if self.use_pn else x
        return x


class Swish(layers.Layer):
    def __init__(self):
        super(Swish, self).__init__()

    def call(self, inputs):
        return inputs * tf.sigmoid(inputs)

class Generator(tf.keras.Model):
    def __init__(self,in_channels, img_channels):
        super(Generator, self).__init__()

        self.init = tf.keras.Sequential([
            PixelNorm(),
            tf.keras.layers.Conv2DTranspose(in_channels, 4, strides=4, padding='same'),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            WSConv2d(in_channels, in_channels, k=3, s=1),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            PixelNorm()
        ])

        self.init_rgb = WSConv2d(in_channels, img_channels, k=1, s=1)
        self.prog_blocks, self.rgb_blocks = [], [self.init_rgb]

        for i in range(len(factors) - 1):
            conv_in_c = int(in_channels * factors[i])
            conv_out_c = int(in_channels * factors[i + 1])
            self.prog_blocks.append(ConvBlock(conv_in_c, conv_out_c))
            self.rgb_blocks.append(WSConv2d(conv_out_c, img_channels, k=1, s=1))
            self.up = tf.keras.layers.UpSampling2D(2)

    def fade_in(self, alpha, upscaled, generated):
        return tf.tanh(alpha * generated + (1 - alpha) * upscaled)

    def call(self, x, alpha, steps):
        out = self.init(x)

        if steps == 0:
            return self.init_rgb(out)

        for step in range(steps):
            upscaled = self.up(out)
            out = self.prog_blocks[step](upscaled)

        final_upscaled = self.rgb_blocks[steps - 1](upscaled)
        final_out = self.rgb_blocks[steps](out)
        return self.fade_in(alpha=alpha, upscaled=final_upscaled, generated=final_out)  # 图像融合

class Discriminator(tf.keras.Model):
    def __init__(self,in_channels,img_channels=3):
        super(Discriminator, self).__init__()
        self.prog_blocks,self.rgb_layers = [],[]
        self.leak = tf.keras.layers.LeakyReLU(0.2)

        for i in range(len(factors)-1,0,-1):
            conv_in_c = int(in_channels * factors[i])
            conv_out_c = int(in_channels * factors[i-1])
            self.prog_blocks.append(ConvBlock(conv_in_c,conv_out_c))
            self.rgb_layers.append(WSConv2d(img_channels,conv_in_c,k=1,s=1))

        self.initial_rgb = WSConv2d(img_channels,in_channels,k=1,s=1)
        self.rgb_layers.append(self.initial_rgb)
        self.avg_pool = tf.keras.layers.AveragePooling2D(2)
        self.minibatch_std = minibatch_std()
        self.final_block = Sequential([
            WSConv2d(in_channels+1,in_channels,k=3,s=1),  # 为什么要加一是因为后面我们要加一个batch
            tf.keras.layers.LeakyReLU(0.2),
            WSConv2d(in_channels, in_channels, k=4, s=1),
            tf.keras.layers.LeakyReLU(0.2),
            WSConv2d(in_channels,1,k=1,s=4),
            tf.keras.layers.Flatten()
        ])

    def fade_in(self, alpha, downscale, generated):
        return tf.tanh(alpha * generated + (1 - alpha) * downscale)

    def call(self,x,alpha,steps):
        cur_step = len(self.prog_blocks) - steps
        out = self.leak(self.rgb_layers[cur_step](x))

        if steps == 0:
            out = self.minibatch_std(out)
            return self.final_block(out) # 将后面进行展开，确保获得[?,1]的数据

        downscale = self.leak(self.rgb_layers[cur_step + 1](self.avg_pool(x)))

        out = self.avg_pool(self.prog_blocks[cur_step](out))
        out = self.fade_in(alpha,downscale,out)

        for step in range(cur_step + 1,len(self.prog_blocks)):
            out = self.prog_blocks[step](out)
            out = self.avg_pool(out)
        out = self.minibatch_std(out)
        return self.final_block(out)
if __name__ == "__main__":
    name = int(1)
    model_g =Discriminator(in_channels=512,img_channels=3)
    x = tf.random.normal([1,1024,1024,3])
    y = model_g(x, alpha=0.5, steps=8)
    model_g.summary()
    model_g.save_weights('1.h5')
    print(y.shape)