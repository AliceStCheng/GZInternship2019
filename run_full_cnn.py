# get tensorflow
import tensorflow as tf

# get functools
import functools

# for image augmentation
from keras.preprocessing.image import ImageDataGenerator

# for the CovNet layers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

# for using a Keras backend
from keras import backend as K

# for regularization and optimization
# (i.e. reduce fluctuations and improve accuracy)
from keras import optimizers
from keras import regularizers
from keras.callbacks import EarlyStopping

# for saving model - you need to use the load model
# command to load it (it's not here)
from keras.models import load_model

# for printing the predicted classification
from sklearn.metrics import classification_report

# for plotting and maths
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

# for plotting the ROC curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from keras.callbacks import Callback
import scikitplot as skplt
from keras import metrics



# dimensions of our images.
img_width, img_height = 150, 150

# ground truth labels, e.g. extsrc, nonextsrc

train_data_dir = '/Users/lancastro/Desktop/Alice/cnn_data/train'
validation_data_dir = '/Users/lancastro/Desktop/Alice/cnn_data/validation'
nb_train_samples = 2000 # total number of files you have in your training set
nb_validation_samples = 800 # total number of files you have in your validation set
epochs = 3 # 300 is a good number, it but needs to run overnight
batch_size = 128

# Seed random numbers
np.random.seed(1337)

# makes Keras go channel first - Google what channels_first means.
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

# the CovNet layers
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64, input_dim=20, kernel_regularizer=regularizers.l2(0.01)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
#model.add(Activation('sigmoid'))

# load model weights from the previous run
#model.load_weights('first_try.h5')

# implements early stopping to prevent overtraining/overfitting
EarlyStopping(monitor='val_err', patience=5)

# define the optimizer, the value that kind of works is lr=0.001.
#1e-4 (taken from the original tutorial) works really well, actually.
opt = optimizers.SGD(lr=0.0001, momentum=0.9)
# compiles the network for training
model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=359,
    horizontal_flip=True,
    vertical_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

# trains the network
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')
# class_dictionary = train_generator.class_indices
# print(train_generator, class_dictionary)
print(train_generator)
# tests using the validation set
validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

# creates the model
history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
    use_multiprocessing=True)

# predicts the outcome
Y_pred = model.predict_generator(
                    validation_generator,
                    steps=nb_validation_samples // batch_size+1, # used to be 5
                    use_multiprocessing=False)
                    #callbacks=[roc_callback(training_data=(X_train, y_train)))
# prints first 10 prediction values
print(Y_pred[0:10])
# returns the position of the largest value,
# so it prints the highest prediction value
y_pred = np.argmax(Y_pred, axis=1)
#fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred)
# saves the model
model.save('classification_model.h5')

# this makes the matplotlib graphs look pretty
style.use('fast')

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

# plots accuracy and validation
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.draw()
plt.savefig('accuracy.png', dpi=300)
#plt.show()

# creates ground truth labels, this is not needed if you're
# not using a generator (small samples size only), but we're
# using a generator here.
y_true = validation_generator.classes
# m = tf.keras.metrics.AUC(num_thresholds=3)
# m.update_state(y_true)
# function for plotting a ROC curve

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#WORK ON THIS NEXT WEEK!!!
#validation_generator = np.reshape()
print('Confusion Matrix')
print("validation_generator.shape is this:")
print(validation_generator.classes.shape) # prints the first of the [,]
print("y_pred.shape is this:")
print(y_pred.shape) # prints the second of the [,]
print("Y_pred.shape is this:")
print(Y_pred.shape) # prints the second of the [,]
#print(confusion_matrix(validation_generator.classes, Y_pred))
target_names = ['extended_sources', 'non_es']
print("validation_generator.class is this:")
print(validation_generator.classes)
print("validation_generator.class_indices is this:")
print(validation_generator.class_indices) # prints the index of each class
#print('Classification Report')
#print(classification_report(validation_generator.classes, y_pred, target_names=target_names))

# tf.metrics.auc(
#     labels,
#     validation_generator.classes
# )
#skplt.metrics.plot_roc_curve(y_true, y_pred)
#plt.show()
