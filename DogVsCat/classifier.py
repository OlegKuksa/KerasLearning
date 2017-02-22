from time import sleep
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import Callback
from keras.optimizers import Nadam, Adam



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


NesAdam = Nadam(lr=0.00025)

model.compile(loss='binary_crossentropy',
              optimizer=NesAdam,
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='binary'
    )

validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='binary'
        )

model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples,
        nb_epoch=nb_epoch,
        validation_data=validation_generator,
        nb_val_samples=nb_validation_samples,
        )

model.save_weights('save_weight.h5')
