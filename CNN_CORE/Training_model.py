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

def get_session(gpu_fraction=0.8):
    '''Assume that you have 24GB of GPU memory and want to allcate ~6GB'''
    
    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
 
    if num_threads:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
 
 
KK.set_session(get_session())

batch_size = 32
epochs = 3

train_X = np.load('../Challenge_numpy/train_X_130k.npy')
#test_X = np.load('../Challenge_numpy/test_X_1k.npy') #대회 때 이것만 바꾼후 테스트 돌리기 
test_X = np.load('../Random10000_numpy/10000_Random_unpack.npy') #대회 때 이것만 바꾼후 테스트 돌리기 
train_Y = np.load('../Challenge_numpy/Y_train_130k.npy')

inputs = Input(shape=(224, 224, 1), dtype = 'float32', name='input')

#훈련된 모델 불러오기
#resnet = load_model('../Challenge_Model/Training_model_Challenge_padding.h5')
#resnet = load_model('../Challenge_Model/Training_model_2020.h5')
resnet = load_model('../Challenge_Model/Training_model_Challenge_padding.h5')

#resnet.compile(optimizer = Adam(lr=0.001 ,beta_1 = 0.9, beta_2 = 0.999), loss='categorical_crossentropy', metrics = ['accuracy'])
resnet.summary()

train_X = train_X.reshape(-1,224, 224,1)
test_X = test_X.reshape(-1,224, 224,1)

datagen = ImageDataGenerator(rotation_range = 10, fill_mode='nearest')
datagen.fit(train_X)

his = resnet.fit_generator(datagen.flow(train_X, train_Y, batch_size = batch_size), epochs = epochs, steps_per_epoch=train_X.shape[0] / batch_size, callbacks=[ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.1)], verbose = 1)  


#라벨 생성
name = []
y_p = []
y_pred = resnet.predict(test_X)
#y_pred = np.where(np.array(y_pred) > 0.5, 1, 0)
for i in range(len(y_pred)):
    y_p.append(y_pred[i][1])

#files = sorted(glob.glob('../datasets/180705_VX_Heaven/*'))
files = sorted(glob.glob('../datasets/10000_Random_unpack/*'))
for file in files:
    path = os.path.dirname(file)
    base_name = os.path.splitext(os.path.basename(file))[0]
    vir = os.path.splitext(os.path.basename(file))[-1]
    base_name = base_name + vir

    name.append(base_name)

print(len(y_p))
print(len(name))
series = OrderedDict([('hash', name), ('y_pred', y_p)])
result = pd.DataFrame.from_dict(series)
#result.to_csv("../results/VX_Heaven_23k", index=False)  
#result.to_csv("../Challenge_result_label/Challenge_Label_2020_2020.csv", index=False)
#result.to_csv("../Challenge_result_label/Challenge_Label_2020_1k.csv", index=False)
result.to_csv("../results/10000_Random_VX.csv", index=False)
