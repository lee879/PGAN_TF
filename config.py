import tensorflow as tf

START_TRAIN_AT_IMG_SIZE = 4
DATASET = r'.\data2'
CHECKPOINT_GEN = "generator.pth"
CHECKPOINT_CRITIC = "critic.pth"
SAVE_MODEL = True
LOAD_MODEL = False
LEARNING_RATE = 1e-3
#BATCH_SIZES = [64, 64, 64,32, 20,4, 4, 4, 1]
BATCH_SIZES = [1024, 512, 256 , 64 , 32 ,8,4,1,1]
CHANNELS_IMG = 3
Z_DIM = 256# should be 512 in original paper 论文中给出的输入是一个1x1x512的latent ，这里了给的是256是因为硬件的问题
IN_CHANNELS = 256 # should be 512 in original paper
CRITIC_ITERATIONS = 1
LAMBDA_GP = 10
PROGRESSIVE_EPOCHS = [1000,2000,3000,4000,10000,15000,30000,40000,60000] # 每个图片训练的epoch
#PROGRESSIVE_EPOCHS = [1,1,1,1,1,1,1,1,1]
FIXED_NOISE = tf.random.normal((8, Z_DIM, 1, 1)) #使用gpu参数随机噪声
NUM_WORKERS = 4 #使用图片迭代器的个数（如果使用的太多会出现内存不足）
Factors_G_Channels = [512,512,512,512,256,128,64,32,16] # 一共有9个模块
Factors_D_Channels = [16,32,64,128,256,512,512,512,512] # 一共有9个模块
Factors_Img_Size = [4,8,16,32,64,128,256,512,1024]
ALPHA = 0.005