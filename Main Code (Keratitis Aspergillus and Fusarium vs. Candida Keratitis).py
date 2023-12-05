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
EP = 250
BS = 15
CW = {0: 25, 1: 1}

FG_dir = './Data/Fungal/'
FG_names = ['Candida']
FG_data = []
for i in range(len(FG_names)):
    filelist = glob.glob(FG_dir + FG_names[i] + '/*.JPG')
    FG_data.append([np.array(Image.open(fname).resize((shape[1], shape[0]))) for fname in filelist])
FG_data = np.array(FG_data)
FG_data = FG_data[0]
FG_out = np.zeros((len(FG_data)))

FA_dir = './Data/Fungal/'
FA_names = ['Aspergillus', 'Fusarium']
FA_data = []
for i in range(len(FA_names)):
    filelist = glob.glob(FA_dir + FA_names[i] + '/*.JPG')
    FA_data.append([np.array(Image.open(fname).resize((shape[1], shape[0]))) for fname in filelist])
FA_data = np.array(FA_data)
FA_data = np.concatenate((FA_data[0], FA_data[1]), axis=0)
FA_out = np.ones((len(FA_data)))

print('Number of Candida and Other Types of Fungal Keratitis Samples:', len(FG_data))
print('Number of Aspergillus and Fusarium Type of Fungal Keratitis Samples:', len(FA_data))

X = np.concatenate((FA_data, FG_data), axis=0)
y = np.concatenate((FA_out, FG_out), axis=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TS, shuffle=True)

inp = Input(shape=shape)

x_1 = Conv2D(1, 5, activation='sigmoid', padding='same')(inp)
x_1 = Dropout(0.1)(x_1)
x_1 = BatchNormalization()(x_1)

x_2 = Conv2D(1, 10, activation='sigmoid', padding='same')(inp)
x_2 = Dropout(0.1)(x_2)
x_2 = BatchNormalization()(x_2)

x_3 = Conv2D(1, 20, activation='sigmoid', padding='same')(inp)
x_3 = Dropout(0.1)(x_3)
x_3 = BatchNormalization()(x_3)

x_concat = tf.concat((x_1, x_2, x_3), axis=3)

x = Conv2D(6, 5, strides=1, activation='relu')(x_concat)
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
heatmap_model = Model(inp, x_concat)
opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics='accuracy')
model.summary()

model.fit(X_train, y_train, validation_split=TS / 4, epochs=EP, batch_size=BS, class_weight=CW)
model.save('./Models/Main Model V3.0')
heatmap_model.save('./Models/Main Model (Heatmap) V3.0')

results = model.evaluate(X_test, y_test)
print('Evaluation Results:', results)
y_pred = model.predict(X_test)
y_pred[y_pred < 0.5] = 0
y_pred[y_pred >= 0.5] = 1
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred, normalize='true'))
