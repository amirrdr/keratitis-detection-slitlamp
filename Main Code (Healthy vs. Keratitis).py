import glob
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.layers import *
from keras import Model
from sklearn.metrics import confusion_matrix

data_dir = './Data/'
shape = (300, 300, 3)
TS = 0.1
EP = 75
BS = 20
CW = {0: 1, 1: 1}

FG_dir = './Data/Fungal/'
FG_names = ['Aspergillus', 'Candida', 'Fusarium', 'Others']
FG_data = []
for i in range(len(FG_names)):
    filelist = glob.glob(FG_dir + FG_names[i] + '/*.JPG')
    FG_data.append([np.array(Image.open(fname).resize((shape[1], shape[0]))) for fname in filelist])
FG_data = np.array(FG_data)
FG_data = np.concatenate((FG_data[0], FG_data[1], FG_data[2], FG_data[3]), axis=0)
FG_out = np.ones((len(FG_data)))

print('Number of Fungal Keratitis Samples:', len(FG_data))

BC_dir = './Data/Bacterial/'
filelist = glob.glob(BC_dir + '*.JPG')
BC_data = [np.array(Image.open(fname).resize((shape[1], shape[0]))) for fname in filelist]
BC_data = np.array(BC_data)
BC_out = np.ones((len(BC_data)))

print('Number of Bacterial Keratitis Samples:', len(BC_data))

HL_dir = './Data/Healthy/'
filelist = glob.glob(HL_dir + '*.JPG')
HL_data = [np.array(Image.open(fname).resize((shape[1], shape[0]))) for fname in filelist]
HL_data = np.array(HL_data)
HL_out = np.zeros((len(HL_data)))

print('Number of Healthy Samples:', len(HL_data))
print('Number of Keratitis Samples:', len(BC_data) + len(FG_data))

X = np.concatenate((HL_data, BC_data, FG_data), axis=0)
y = np.concatenate((HL_out, BC_out, FG_out), axis=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TS, shuffle=True)

inp = Input(shape=shape)

x_1 = Conv2D(1, 5, activation='relu', padding='same')(inp)
x_1 = Dropout(0.1)(x_1)
x_1 = BatchNormalization()(x_1)

x_2 = Conv2D(1, 10, activation='relu', padding='same')(inp)
x_2 = Dropout(0.1)(x_2)
x_2 = BatchNormalization()(x_2)

x_3 = Conv2D(1, 20, activation='relu', padding='same')(inp)
x_3 = Dropout(0.1)(x_3)
x_3 = BatchNormalization()(x_3)

x = tf.concat((x_1, x_2, x_3), axis=3)

x = Conv2D(6, 5, strides=1, activation='relu')(x)
x = Dropout(0.1)(x)
x = BatchNormalization()(x)

x = Conv2D(3, 10, strides=1, activation='relu')(x)
x = Dropout(0.1)(x)
x = BatchNormalization()(x)

x = Conv2D(1, 20, strides=1, activation='relu')(x)
x = Dropout(0.1)(x)
x = BatchNormalization()(x)

x = Flatten()(x)
x = Dense(25, activation='relu')(x)
x = Dropout(0.1)(x)
x = BatchNormalization()(x)
x = Dense(10, activation='relu')(x)
x = Dropout(0.1)(x)
out = Dense(1, activation='sigmoid')(x)

model = Model(inp, out)
opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics='accuracy')
model.summary()

model.fit(X_train, y_train, validation_split=TS / 4, epochs=EP, batch_size=BS, class_weight=CW)
model.save('./Models/Main Model V1.0')

results = model.evaluate(X_test, y_test)
print('Evaluation Results:', results)
y_pred = model.predict(X_test)
y_pred[y_pred < 0.5] = 0
y_pred[y_pred >= 0.5] = 1
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred, normalize='true'))
