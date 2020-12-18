import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras import layers
from keras import Model
from keras.models import Sequential
from keras.layers import Input, Dense, Activation, Flatten, Conv2D, MaxPooling2D, Add, GlobalAveragePooling2D, ZeroPadding2D, AveragePooling2D, LeakyReLU
from keras.layers import BatchNormalization
from keras.optimizers import Adam, SGD, Nadam, Adamax
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
import keras.backend.tensorflow_backend as KK
from keras.models import load_model
from collections import OrderedDict
import glob
import pandas as pd

#from sub_model import *
#watch -n 1 nvidia-smi

#데이터 로드 
train_X = np.load('../results/train_X_2020.npy')
test_X = np.load('../results/test_X_2020.npy')
train_Y = np.load('../results/Y_train_2020.npy')
test_Y = np.load('../results/Y_test_2020.npy')


def get_session(gpu_fraction=0.8):
    '''Assume that you have 24GB of GPU memory and want to allcate ~6GB'''
    
    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
 
    if num_threads:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
 
 
KK.set_session(get_session())

#model = []
batch_size = 32
epochs = 50
learning_rate = 0.001

#입력 데이터 
inputs = Input(shape=(224, 224,1), dtype='float32', name='input')

#Resnet50
def conv1(x):
    #Model = Sequential()
    #Layer1    
    x = ZeroPadding2D(padding=(3,3))(x)
    x = Conv2D(32, (3,3), strides=(2, 2))(x) # output = (112, 112, 64)
    x = ZeroPadding2D(padding=(1,1))(x)
    x = Conv2D(32, (3,3), strides=(1, 1))(x)
    x = Conv2D(64, (3,3), strides=(1, 1))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #x = LeakyReLU(0.3)(x)
    x = ZeroPadding2D(padding=(1,1))(x)
    return x

def conv2(x):
    #Layer2
    x = MaxPooling2D((3,3),2)(x) # output = (56, 56, 64)
    short_cut = x
    for i in range(3):
       if(i == 0):
          x = Conv2D(filters = 64, kernel_size = (1, 1), strides = (1,1), padding = 'valid')(x) # output = (56, 56, 64)
          x = BatchNormalization()(x)
          x = Activation('relu')(x)
          #x = LeakyReLU(0.3)(x)

          x = Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
          x = BatchNormalization()(x)
          x = Activation('relu')(x)
          #x = LeakyReLU(0.3)(x)

          x = Conv2D(filters = 256, kernel_size = (1, 1), strides = (1,1), padding = 'valid')(x) 
          short_cut = Conv2D(filters = 256, kernel_size = (1, 1), strides = (1,1), padding = 'valid')(short_cut) # output = (56, 56, 64)
          x = BatchNormalization()(x)
          short_cut = BatchNormalization()(short_cut)
    
          x = Add()([x, short_cut])
          x = Activation('relu')(x)
          #x = LeakyReLU(0.3)(x)
          short_cut = x
       else:
          x = Conv2D(filters = 64, kernel_size = (1, 1), strides = (1,1), padding = 'valid')(x)
          x = BatchNormalization()(x)
          x = Activation('relu')(x)
          #x = LeakyReLU(0.3)(x)

          x = Conv2D(filters = 64, kernel_size = (3, 3), strides = (1,1), padding = 'same')(x)
          x = BatchNormalization()(x)
          x = Activation('relu')(x)
          #x = LeakyReLU(0.3)(x)

          x = Conv2D(filters = 256, kernel_size = (1, 1), strides = (1,1), padding = 'valid')(x)
          x = BatchNormalization()(x)

          x = Add()([x, short_cut])
          x = Activation('relu')(x)
          #x = LeakyReLU(0.3)(x)
          short_cut = x
    return x

def conv3(x):
    # input = (56, 56, 128)
    short_cut = x
    for i in range(4):
       if(i == 0):
          x = Conv2D(filters = 128, kernel_size = (1, 1), strides = (1, 1), padding = 'valid')(x)
          x = BatchNormalization()(x)
          x = Activation('relu')(x)
          #x = LeakyReLU(0.3)(x)

          x = Conv2D(filters = 128, kernel_size = (3, 3), strides = (2, 2), padding = 'same')(x)   # output = (28, 28, 128)
          x = BatchNormalization()(x)
          x = Activation('relu')(x)
          #x = LeakyReLU(0.3)(x)

          x = Conv2D(filters = 512, kernel_size = (1, 1), strides = (1, 1), padding = 'valid')(x) # output = (28, 28, 512)
          short_cut = AveragePooling2D((2,2),2)(short_cut)  # output = (28, 28, 128)
          short_cut = Conv2D(filters = 512, kernel_size = (1, 1), strides = (1,1), padding = 'valid')(short_cut)
          x = BatchNormalization()(x)
          short_cut = BatchNormalization()(short_cut)

          x = Add()([x, short_cut])
          x = Activation('relu')(x)
          #x = LeakyReLU(0.3)(x)
          short_cut = x
       else:
          x = Conv2D(filters = 128, kernel_size = (1, 1), strides = (1, 1), padding = 'valid')(x)
          x = BatchNormalization()(x)
          x = Activation('relu')(x)
          #x = LeakyReLU(0.3)(x)

          x = Conv2D(filters = 128, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
          x = BatchNormalization()(x)
          x = Activation('relu')(x)
          #x = LeakyReLU(0.3)(x)

          x = Conv2D(filters = 512, kernel_size = (1, 1), strides = (1, 1), padding = 'valid')(x)
          x = BatchNormalization()(x)

          x = Add()([x, short_cut])
          x = Activation('relu')(x)
          #x = LeakyReLU(0.3)(x)
          short_cut = x
    return x

def conv4(x):
    # input = (28, 28, 512)
    short_cut = x
    for i in range(6):
       if(i == 0):
          x = Conv2D(filters = 256, kernel_size = (1, 1), strides = (1, 1), padding = 'valid')(x)
          x = BatchNormalization()(x)
          x = Activation('relu')(x)
          #x = LeakyReLU(0.3)(x)

          x = Conv2D(filters = 256, kernel_size = (3, 3), strides = (2, 2), padding = 'same')(x)  # output = (14, 14, 512)
          x = BatchNormalization()(x)
          x = Activation('relu')(x)
          #x = LeakyReLU(0.3)(x)

          x = Conv2D(filters = 1024, kernel_size = (1, 1), strides = (1, 1), padding = 'valid')(x)
          short_cut = AveragePooling2D((2,2),2)(short_cut) # output = (14, 14, 512)
          short_cut = Conv2D(filters = 1024, kernel_size = (1, 1), strides = (1,1), padding = 'valid')(short_cut)
          x = BatchNormalization()(x)
          short_cut = BatchNormalization()(short_cut)

          x = Add()([x, short_cut])
          x = Activation('relu')(x)
          #x = LeakyReLU(0.3)(x)
          short_cut = x

       else:
          x = Conv2D(filters = 256, kernel_size = (1, 1), strides = (1, 1), padding = 'valid')(x)
          x = BatchNormalization()(x)
          x = Activation('relu')(x)
          #x = LeakyReLU(0.3)(x)

          x = Conv2D(filters = 256, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
          x = BatchNormalization()(x)
          x = Activation('relu')(x)
          #x = LeakyReLU(0.3)(x)

          x = Conv2D(filters = 1024, kernel_size = (1, 1), strides = (1, 1), padding = 'valid')(x)
          x = BatchNormalization()(x)

          x = Add()([x, short_cut])
          x = Activation('relu')(x)
          #x = LeakyReLU(0.3)(x)
          short_cut = x
    return x

def conv5(x):
    #input = (14, 14, 512)
    short_cut = x
    for i in range(3):
       if(i == 0):
          x = Conv2D(filters = 512, kernel_size = (1, 1), strides = (1, 1), padding = 'valid')(x)
          x = BatchNormalization()(x)
          x = Activation('relu')(x)
          #x = LeakyReLU(0.3)(x)

          x = Conv2D(filters = 512, kernel_size = (3, 3), strides = (2, 2), padding = 'same')(x)
          x = BatchNormalization()(x)
          x = Activation('relu')(x)
          #x = LeakyReLU(0.3)(x)

          x = Conv2D(filters = 2048, kernel_size = (1, 1), strides = (1, 1), padding = 'valid')(x)
          short_cut = AveragePooling2D((2,2),2)(short_cut)
          short_cut = Conv2D(filters = 2048, kernel_size = (1, 1), strides = (1,1), padding = 'valid')(short_cut)
          x = BatchNormalization()(x)
          short_cut = BatchNormalization()(short_cut)

          x = Add()([x, short_cut])
          x = Activation('relu')(x)
          #x = LeakyReLU(0.3)(x)
          short_cut = x

       else:
          x = Conv2D(filters = 512, kernel_size = (1, 1), strides = (1, 1), padding = 'valid')(x)
          x = BatchNormalization()(x)
          x = Activation('relu')(x)
          #x = LeakyReLU(0.3)(x)

          x = Conv2D(filters = 512, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
          x = BatchNormalization()(x)
          x = Activation('relu')(x)
          #x = LeakyReLU(0.3)(x)

          x = Conv2D(filters = 2048, kernel_size = (1, 1), strides = (1, 1), padding = 'valid')(x)
          x = BatchNormalization()(x)

          x = Add()([x, short_cut])
          x = Activation('relu')(x)
          #x = LeakyReLU(0.3)(x)
          short_cut = x
    return x

x = conv1(inputs)
x = conv2(x)
x = conv3(x)
x = conv4(x)
x = conv5(x)

x = GlobalAveragePooling2D()(x)

outputs = Dense(2, activation='softmax')(x)

#모델 생성 
resnet = Model(inputs, outputs)

resnet.summary()
#decay_rate = 0.00495
#if learning_rate < 0.001:
#    decay_rate = 0
resnet.compile(optimizer = Adam(lr=learning_rate ,beta_1 = 0.9, beta_2 = 0.999, amsgrad=True), loss='categorical_crossentropy', metrics = ['accuracy'])
#resnet.compile(optimizer = SGD(lr=learning_rate ,momentum = momentum, decay = decay_rate, nesterov = False), loss='categorical_crossentropy', metrics = ['accuracy'])
#resnet.compile(optimizer = Nadam(lr=0.002 ,beta_1 = 0.9, beta_2 = 0.999), loss='categorical_crossentropy', metrics = ['accuracy'])
#resnet.compile(optimizer = Adamax(lr=0.002, beta_1 = 0.9, beta_2 = 0.999), loss='categorical_crossentropy', metrics = ['accuracy'])

train_X = train_X.reshape(-1,224, 224,1)
test_X = test_X.reshape(-1,224, 224,1)

#이미지 Generator -> 이미지를 변형시켜서 실시간으로 늘려줌
datagen = ImageDataGenerator(rotation_range = 10, fill_mode='nearest')
datagen.fit(train_X)

his = resnet.fit_generator(datagen.flow(train_X, train_Y, batch_size = batch_size), epochs = epochs, steps_per_epoch=train_X.shape[0] / batch_size, validation_data = (test_X, test_Y), callbacks=[ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.1)], verbose = 1)  

#모델 저장
"""
model_json = resnet.to_json()
with open("../results/ResNet50_Challenge_All.json","w") as json_file:
    json_file.write(model_json)

#가중치 저장
resnet.save_weights("../results/ResNet50_Weight_Challenge_All.h5")
"""
#전체 저장
resnet.save('../results/Training_model_2020.h5')
print("Save model")

#라벨 생성
name = []
y_p = []
y_pred = resnet.predict(test_X)
#y_pred = np.where(np.array(y_pred) > 0.4, 1, 0)
for i in range(len(y_pred)):
    y_p.append(y_pred[i][1])

files = sorted(glob.glob('../datasets/2020/*'))
for file in files:
    path = os.path.dirname(file)
    base_name = os.path.splitext(os.path.basename(file))[0]
    base_name = base_name + "vir"

    name.append(base_name)

series = OrderedDict([('hash', name), ('y_pred', y_p)])
result = pd.DataFrame.from_dict(series)
result.to_csv("../results/result_2020_2020_2020.csv", index=False)  
print("Label Save")
#결과 그래프
fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()

loss_ax.plot(his.history['accuracy'], 'y', label='acc')
loss_ax.plot(his.history['val_accuracy'], 'r', label='val_acc')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('acc')
loss_ax.legend(loc = 'upper left')

plt.show()


    
