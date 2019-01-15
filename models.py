import numpy as np
from keras.layers import Conv2D, BatchNormalization, LeakyReLU, Dropout, Input, MaxPooling2D, Conv2DTranspose,Lambda
from keras.layers import Concatenate
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.losses import binary_crossentropy
from keras.preprocessing.image import ImageDataGenerator
import keras.callbacks as CallBacks
from keras import backend as K
import tensorflow as tf





def unet(in_shape=(256,256,3), alpha=0.1, dropout=None):

    #    dropout = [0.1,0.2,0.25,0.3,0.5]
    #  ------ model definition -----
    Unet_Input = Input(shape=in_shape)
    # segment no. 1 --- starting encoder part
    conv1_1  = Conv2D(32, kernel_size=(3,3), strides = (1,1), padding = 'same')(Unet_Input)
    relu1_1  = LeakyReLU(alpha = alpha)(conv1_1)
    conv1_2  = Conv2D(32, kernel_size=(3,3), strides = (1,1), padding = 'same')(relu1_1)
    relu1_2  = LeakyReLU(alpha = alpha)(conv1_2)    
    bn1      =  BatchNormalization()(relu1_2)
    maxpool1 = MaxPooling2D(pool_size = (2,2), strides = (2,2))(bn1)    
    # segment no. 2
    conv2_1  = Conv2D(64, kernel_size=(3,3), strides = (1,1), padding = 'same')(maxpool1)
    relu2_1  = LeakyReLU(alpha = alpha)(conv2_1)    
    conv2_2  = Conv2D(64, kernel_size=(3,3), strides = (1,1), padding = 'same')(relu2_1)
    relu2_2  = LeakyReLU(alpha = alpha)(conv2_2)    
    bn2      =  BatchNormalization()(relu2_2)
    maxpool2 = MaxPooling2D(pool_size = (2,2), strides = (2,2))(bn2)    
    # segment no. 3
    conv3_1  = Conv2D(128, kernel_size=(3,3), strides = (1,1), padding = 'same')(maxpool2)
    relu3_1  = LeakyReLU(alpha = alpha)(conv3_1)    
    conv3_2  = Conv2D(128, kernel_size=(3,3), strides = (1,1), padding = 'same')(relu3_1)
    relu3_2  = LeakyReLU(alpha = alpha)(conv3_2)    
    bn3      =  BatchNormalization()(relu3_2)
    maxpool3 = MaxPooling2D(pool_size = (2,2), strides = (2,2))(bn3)    
    # segment no. 4
    conv4_1  = Conv2D(256, kernel_size=(3,3), strides = (1,1), padding = 'same')(maxpool3)
    relu4_1  = LeakyReLU(alpha = alpha)(conv4_1)    
    conv4_2  = Conv2D(256, kernel_size=(3,3), strides = (1,1), padding = 'same')(relu4_1)
    relu4_2  = LeakyReLU(alpha = alpha)(conv4_2)    
    bn4      =  BatchNormalization()(relu4_2)
    maxpool4 = MaxPooling2D(pool_size = (2,2), strides = (2,2))(bn4)        
    # segment no. 5 --- start of decoder part
    conv5_1  = Conv2DTranspose(256, kernel_size=(3,3), strides = (2,2), padding = 'same')(maxpool4)
    relu5_1  = LeakyReLU(alpha = alpha)(conv5_1)
    conc5    = Concatenate(axis=3)([relu5_1, relu4_2])
    conv5_2  = Conv2D(256, kernel_size=(3,3), strides = (1,1), padding = 'same')(conc5)
    relu5_2  = LeakyReLU(alpha = alpha)(conv5_2)
    bn5      = BatchNormalization()(relu5_2)
    # segment no. 6
    conv6_1  = Conv2DTranspose(128, kernel_size=(3,3), strides = (2,2), padding = 'same')(bn5)
    relu6_1  = LeakyReLU(alpha = alpha)(conv6_1)
    conc6    = Concatenate(axis=3)([relu6_1, relu3_2])
    conv6_2  = Conv2D(128, kernel_size=(3,3), strides = (1,1), padding = 'same')(conc6)
    relu6_2  = LeakyReLU(alpha = alpha)(conv6_2)
    bn6      = BatchNormalization()(relu6_2)
    # segment no. 7
    conv7_1  = Conv2DTranspose(64, kernel_size=(3,3), strides = (2,2), padding = 'same')(bn6)
    relu7_1  = LeakyReLU(alpha = alpha)(conv7_1)
    conc7    = Concatenate(axis=3)([relu7_1, relu2_2])
    conv7_2  = Conv2D(64, kernel_size=(3,3), strides = (1,1), padding = 'same')(conc7)
    relu7_2  = LeakyReLU(alpha = alpha)(conv7_2)
    bn7      = BatchNormalization()(relu7_2)
    # segment no. 8
    conv8_1  = Conv2DTranspose(32, kernel_size=(3,3), strides = (2,2), padding = 'same')(bn7)
    relu8_1  = LeakyReLU(alpha = alpha)(conv8_1)
    conc8    = Concatenate(axis=3)([relu8_1, relu1_2])
    conv8_2  = Conv2D(32, kernel_size=(3,3), strides = (1,1), padding = 'same')(conc8)
    relu8_2  = LeakyReLU(alpha = alpha)(conv8_2)
    Unet_Output = Conv2D(1, kernel_size=(1,1), strides = (1,1), padding='same', activation='sigmoid')(relu8_2)
    # model
    Unet = Model(Unet_Input, Unet_Output)
    return Unet



def unet_kmeans(in_shape=(256,256,7), alpha=0.1, dropout=None):

    # drop_rate = [0.1,0.2,0.25,0.3,0.5]
    # ---- model definition ----
    Unet_Input       = Input(shape=in_shape)
    Unet_ClusterMaps = Lambda(lambda Unet_Input : Unet_Input[:,:,:,3:])(Unet_Input)
    # segment no. 1 --- start of encoder
    conv1_1  = Conv2D(32, kernel_size=(5,5), strides = (1,1), padding = 'same')(Unet_Input)
    bn1_1    = BatchNormalization()(conv1_1)
    relu1_1  = LeakyReLU(alpha = alpha)(bn1_1)
    conv1_2  = Conv2D(32, kernel_size=(5,5), strides = (1,1), padding = 'same')(relu1_1)
    bn1_2    = BatchNormalization()(conv1_2)
    relu1_2  = LeakyReLU(alpha = alpha)(bn1_2)
    maxpool1 = MaxPooling2D(pool_size = (2,2), strides = (2,2))(relu1_2)
    # segment no. 2
    conv2_1  = Conv2D(64, kernel_size=(3,3), strides = (1,1), padding = 'same')(maxpool1)
    bn2_1    = BatchNormalization()(conv2_1)
    relu2_1  = LeakyReLU(alpha = alpha)(bn2_1)
    conv2_2  = Conv2D(64, kernel_size=(3,3), strides = (1,1), padding = 'same')(relu2_1)
    bn2_2    = BatchNormalization()(conv2_2)
    relu2_2  = LeakyReLU(alpha = alpha)(bn2_2)
    maxpool2 = MaxPooling2D(pool_size = (2,2), strides = (2,2))(relu2_2)
    # segment no. 3
    conv3_1  = Conv2D(128, kernel_size=(3,3), strides = (1,1), padding = 'same')(maxpool2)
    bn3_1    = BatchNormalization()(conv3_1)
    relu3_1  = LeakyReLU(alpha = alpha)(bn3_1)
    conv3_2  = Conv2D(128, kernel_size=(3,3), strides = (1,1), padding = 'same')(relu3_1)
    bn3_2    = BatchNormalization()(conv3_2)
    relu3_2  = LeakyReLU(alpha = alpha)(bn3_2)
    maxpool3 = MaxPooling2D(pool_size = (2,2), strides = (2,2))(relu3_2)
    # segment no. 4
    conv4_1  = Conv2D(256, kernel_size=(3,3), strides = (1,1), padding = 'same')(maxpool3)
    bn4_1    = BatchNormalization()(conv4_1)
    relu4_1  = LeakyReLU(alpha = alpha)(bn4_1)
    conv4_2  = Conv2D(256, kernel_size=(3,3), strides = (1,1), padding = 'same')(relu4_1)
    bn4_2    = BatchNormalization()(conv4_2)
    relu4_2  = LeakyReLU(alpha = alpha)(bn4_2)
    maxpool4 = MaxPooling2D(pool_size = (2,2), strides = (2,2))(relu4_2)    
    # segment no. 5 --- start of decoder
    conv5_1  = Conv2DTranspose(256, kernel_size=(3,3), strides = (2,2), padding = 'same')(maxpool4)
    bn5_1    = BatchNormalization()(conv5_1)
    relu5_1  = LeakyReLU(alpha = alpha)(bn5_1)
    conc5    = Concatenate(axis=3)([relu5_1, relu4_2])
    conv5_2  = Conv2D(256, kernel_size=(3,3), strides = (1,1), padding = 'same')(conc5)
    bn5_2    = BatchNormalization()(conv5_2)
    relu5_2  = LeakyReLU(alpha = alpha)(bn5_2)
    # segment no. 6
    conv6_1 = Conv2DTranspose(128, kernel_size=(3,3), strides = (2,2), padding = 'same')(relu5_2)
    bn6_1   = BatchNormalization()(conv6_1)
    relu6_1 = LeakyReLU(alpha = alpha)(bn6_1)
    conc6   = Concatenate(axis=3)([relu6_1, relu3_2])
    conv6_2 = Conv2D(128, kernel_size=(3,3), strides = (1,1), padding = 'same')(conc6)
    bn6_2   = BatchNormalization()(conv6_2)
    relu6_2 = LeakyReLU(alpha = alpha)(bn6_2)
    # segment no. 7
    conv7_1 = Conv2DTranspose(64, kernel_size=(3,3), strides = (2,2), padding = 'same')(relu6_2)
    bn7_1   = BatchNormalization()(conv7_1)
    relu7_1 = LeakyReLU(alpha = alpha)(bn7_1)
    conc7   = Concatenate(axis=3)([relu7_1, relu2_2])
    conv7_2 = Conv2D(64, kernel_size=(3,3), strides = (1,1), padding = 'same')(conc7)
    bn7_2   = BatchNormalization()(conv7_2)
    relu7_2 = LeakyReLU(alpha = alpha)(bn7_2)
    # segment no. 8
    conv8_1 = Conv2DTranspose(32, kernel_size=(3,3), strides = (2,2), padding = 'same')(relu7_2)
    bn8_1   = BatchNormalization()(conv8_1)
    relu8_1 = LeakyReLU(alpha = alpha)(bn8_1)
    conc8   = Concatenate(axis=3)([relu8_1, relu1_2, Unet_ClusterMaps])
    conv8_2 = Conv2D(32, kernel_size=(5,5), strides = (1,1), padding = 'same')(conc8)
    relu8_2 = LeakyReLU(alpha = alpha)(conv8_2)
    Unet_Output = Conv2D(1, kernel_size=(1,1), strides = (1,1), padding='same', activation='sigmoid')(relu8_2)
    # keras model
    Unet = Model( Unet_Input, Unet_Output)
    return Unet


def simple_model():
    input_  = Input(shape=(64,64,3))
    layer   = Conv2D(filters=32, kernel_size=(15,15), strides=(1,1), padding="same", activation="relu")(input_)
    layer   = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same", activation="relu")(layer)
    layer   = Conv2D(filters=32, kernel_size=(4,4), strides=(1,1), padding="same", activation="relu")(layer)
    layer   = Conv2D(filters=64, kernel_size=(5,5), strides=(1,1), padding="same", activation="relu")(layer)
    layer   = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same", activation="relu")(layer)
    layer   = Conv2D(filters=64, kernel_size=(5,5), strides=(1,1), padding="same", activation="relu")(layer)
    output_ = Conv2D(filters=1,  kernel_size=(1,1), strides=(1,1), padding="same", activation="sigmoid")(layer)
    return Model(input_, output_)
