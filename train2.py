from math import log2
import numpy as np
from tensorflow.python.keras.callbacks import ModelCheckpoint
from model_net_1 import Discriminator,Generator
import config
import tensorflow as tf
import glob
from dataset import make_anime_dataset
import os
import math

# 使用特殊的余弦退火函数
def special_cosine(epoch, max_epochs):
    alpha = 1e-5 + (1 - 1e-5) * (1 + math.cos(math.pi * epoch / max_epochs)) / 2
    return min(alpha, 1)
def generate_big_image(image_data,img_size=2):
    # 将前25张图片拼接成一张大图
    rows =img_size
    cols = img_size
    channels = 3
    image_size = image_data.shape[2]
    big_image = np.zeros((rows * image_size, cols * image_size, channels))
    for i in range(rows):
        for j in range(cols):
            big_image[i * image_size:(i + 1) * image_size, j * image_size:(j + 1) * image_size, :] = image_data[
                i * cols + j]
    # 转换为0-255的像素值
    big_image = ((big_image + 1) / 2) * 255
    big_image = big_image.astype(np.uint8)

    return np.expand_dims(big_image,axis=0)

def gradient_penalty(discriminator,batch_xy,fake_image,alpha,steps): # wgan主要的贡献
    #t = tf.random.uniform(batch_xy.shape,minval=0,maxval=1)
    t = tf.random.normal(batch_xy.shape, mean=0., stddev=1.)
    #t = tf.random.uniforml(batch_xy.shape,minval=-1,maxval=1)
    interplate = t * batch_xy + (1 - t) * fake_image
    with tf.GradientTape() as tape:
        tape.watch([interplate])
        d_interplate_logits = discriminator(interplate,alpha,steps)
    grads = tape.gradient(d_interplate_logits,interplate)
    #grads[b,h,w,c]
    grads = tf.reshape(grads,[grads.shape[0],-1])#来进行一个打平的操作
    gp = tf.norm(grads,axis=1)
    gp = tf.reduce_mean((gp - 1) ** 2)
    return gp

def d_loss_fn(init_G, init_D, real_img, laten,alpha,steps):
    fake = init_G(laten,alpha,steps)
    critic_real = init_D(real_img,alpha,steps)
    critic_fake = init_D(tf.stop_gradient(fake),alpha,steps)
    gp = gradient_penalty(init_D,real_img,fake,alpha,steps)
    loss = (
        -(tf.reduce_mean(critic_real) - tf.reduce_mean(critic_fake))
        + config.LAMBDA_GP * gp
        + (0.001 * tf.reduce_mean(critic_real) ** 2)
    )

    return loss,gp
def g_loss_fn(init_G, init_D, laten,alpha,step):
    fake_img = init_G(laten,alpha,step)
    d_fake_logits = init_D(fake_img,alpha,step)
    loss = (-tf.reduce_mean(d_fake_logits))
    return loss,fake_img

def main():
    name_number = 1
    tensorboard_step = 1
    img_size_end = 2
    alpha = config.ALPHA
    summary_writer = tf.summary.create_file_writer(r".\log")
    tf.random.set_seed(666)

    # data
    img_path = glob.glob(r".\data\anime_faces\*.png")
    # dataset, img_shape, _ = make_anime_dataset(img_path, batch_size=config.BATCH_SIZES[0],resize=config.Factors_Img_Size[0])  # 自己建立的数据划分
    # dataset = dataset.repeat()
    # db_iter = iter(dataset)


    checkpoint_dir = './model'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Define the checkpoint to save the model with the best weights based on validation loss
    best_weights_checkpoint_path_d = os.path.join(checkpoint_dir, 'best_d_{}.h5'.format(config.Factors_Img_Size[0])) # 模型保存路径
    best_weights_checkpoint_path_g = os.path.join(checkpoint_dir, 'best_g_{}.h5'.format(config.Factors_Img_Size[0]))

    best_weights_checkpoint_g = ModelCheckpoint(best_weights_checkpoint_path_g,
                                              monitor='loss',
                                              verbose=1,
                                              save_best_only=True,
                                              save_weights_only=True,
                                              mode='min')
    best_weights_checkpoint_d = ModelCheckpoint(best_weights_checkpoint_path_d,
                                                monitor='loss',
                                                verbose=1,
                                                save_best_only=True,
                                                save_weights_only=True,
                                                mode='min')
    step = int(log2(config.START_TRAIN_AT_IMG_SIZE / 4))

    for num_epochs in config.PROGRESSIVE_EPOCHS[step:]:
        # 数据集的大小
        model_G = Generator(config.Z_DIM, 3)
        model_D = Discriminator(config.Z_DIM)

        dataset, img_shape, _ = make_anime_dataset(img_path, batch_size=config.BATCH_SIZES[step], resize=config.Factors_Img_Size[step])  # 自己建立的数据划分
        dataset = dataset.repeat()
        db_iter = iter(dataset)

        for epoch in range(num_epochs):

            alpha = special_cosine(epoch,num_epochs)
            print("epoch",epoch,"num_epochs",num_epochs,"alpha",float(alpha))
            real_img = next(db_iter)
            laten = tf.random.normal([config.BATCH_SIZES[step],1,1,config.Z_DIM], mean=0., stddev=1.) #
            #laten2 = tf.random.normal([config.BATCH_SIZES[step], 1, 1, config.Z_DIM], mean=0., stddev=1.)
            #鉴别器的训练

            with tf.GradientTape() as tape:
                d_loss, gp= d_loss_fn(model_G, model_D, real_img, laten,alpha,step)
            grads = tape.gradient(d_loss,model_D.trainable_variables)
            tf.optimizers.Adam(learning_rate=0.001, beta_1=0,beta_2=0.99).apply_gradients(
                zip(grads, model_D.trainable_variables))
            #生产器的
            with tf.GradientTape() as tape:
                g_loss,fake_img  = g_loss_fn(model_G, model_D,laten,alpha,step)
            grads1 = tape.gradient(g_loss, model_G.trainable_variables)
            tf.optimizers.Adam(learning_rate=0.001, beta_1=0,beta_2=0.99).apply_gradients(zip(grads1, model_G.trainable_variables))

            print(tensorboard_step, "d_loss:", float(d_loss), "g_loss", float(g_loss),"alpha",float(alpha))

            if epoch % 2 == 0:
                with summary_writer.as_default():
                    tf.summary.scalar('d_loss', float(d_loss), step=tensorboard_step)
                    tf.summary.scalar('g_loss', float(g_loss), step=tensorboard_step)
                    img1 = generate_big_image(fake_img,img_size_end)
                    tf.summary.image("fake_image", img1, step=tensorboard_step)
                    img2 = generate_big_image(real_img,img_size_end)
                    tf.summary.image("real_image", img2, step=tensorboard_step)
            tensorboard_step += 1

            if epoch % 100 == 0:
                # for i, w in enumerate(model_G.weights):
                #     print(i, w.name)
                model_G.save_weights(best_weights_checkpoint_path_g)
                print("______________________________________________________")
                # for i, w in enumerate(model_D.weights):
                #     print(i, w.name)
                # model_D.save_weights(best_weights_checkpoint_path_d)
                print("save model no:{}".format(epoch))
        step += 1 # 进行下一次的训练
        try:
            print("当前输出图片大小{}".format(config.Factors_Img_Size[step]))
        except:
            print("结束训练")
            break
        name_number += 100 # name缓存区间

        #将训练好生成器的模型冻结
        # for layer in model_G.layers:
        #     layer.trainable = False
        #model_G = G_Backbone(model_G,config.Factors_G_Channels[step],name=name_number)

        #将训练好鉴别器的模型冻结
        # for layer in model_D.layers:
        #     layer.trainable = False
        #model_D = D_Backbone(model_D,config.Factors_D_Channels[8-step],name=name_number)
        # progan网络中4090的显卡只能训练并生成一张1024 x 1024 的图片
        if step == 7:
            img_size_end=1

main()



