# load and evaluate a saved model
from numpy import loadtxt
from keras.models import load_model

# for regularization and optimization
# (i.e. reduce fluctuations and improve accuracy)
from keras import optimizers
from keras import regularizers
from keras.callbacks import EarlyStopping

# for plotting and maths
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

# for image augmentation
from keras.preprocessing.image import ImageDataGenerator

# dimensions of our images.
img_width, img_height = 150, 150

train_data_dir = '/Users/lancastro/Desktop/Alice/cnn_data/train'
validation_data_dir = '/Users/lancastro/Desktop/Alice/cnn_data/validation'
nb_train_samples = 2000 # total number of files you have in your training set
nb_validation_samples = 800 # total number of files you have in your validation set
epochs = 3 # 300 is a good number, it but needs to run overnight
batch_size = 128

# Seed random numbers
np.random.seed(1337)

# load model
model = load_model('classification_model.h5')
# summarize model.
model.summary()

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=359,
    horizontal_flip=True,
    vertical_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)
# define the optimizer, the value that kind of works is lr=0.001.
#1e-4 (taken from the original tutorial) works really well, actually.
opt = optimizers.SGD(lr=0.0001, momentum=0.9)

# compiles the network for training
model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

# trains the network
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')
# predicts the outcome

Y_pred = model.predict_generator(
                    validation_generator,
                    steps=nb_validation_samples // batch_size+1, # used to be 5
                    use_multiprocessing=False)
                    #callbacks=[roc_callback(training_data=(X_train, y_train)))

# returns the position of the largest value,
# so it prints the highest prediction value
y_pred = np.argmax(Y_pred, axis=1)

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)


# prints first 100 prediction values
print(Y_pred[0:100])

# plots accuracy and validation
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.draw()
plt.savefig('accuracy.png', dpi=300)
#plt.show()
