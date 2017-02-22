import numpy as np
import scipy.misc
from time import sleep
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import Callback
from keras.optimizers import Nadam, Adam
from keras import backend as K


def load_and_scale_imgs():
   img_names = ['data/test/15.jpg',
                'data/test/16.jpg',
                'data/test/2.jpg',
                'data/test/images.jpeg',
                'data/test/ex.jpg',
                'data/test/ex2.jpeg',]
 
   imgs = [np.transpose(scipy.misc.imresize(scipy.misc.imread(img_name), (150, 150)),
                        (2, 0, 1)).astype('float32')
           for img_name in img_names]
   return np.array(imgs) / 255



img_width, img_height = 150, 150

train_data_dir = 'train'
validation_data_dir = 'validation'
nb_train_samples = 2000
nb_validation_samples = 800
nb_epoch = 60


model = Sequential()
model.add(Convolution2D(32, 3, 3, 
                        input_shape=(3, img_width, img_height),
                        init='glorot_normal'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2),
                       dim_ordering="tf"))


model.add(Convolution2D(32, 5, 5, init='glorot_normal'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="tf"))


model.add(Convolution2D(32, 5, 5, init='glorot_normal'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="tf"))




model.add(Convolution2D(32, 7, 7, init='glorot_normal'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="tf"))


model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))



model.load_weights('save_weight_86_percent.h5')


imgs = load_and_scale_imgs()
pred = model.predict_classes(imgs)
print(pred)
K.clear_session()