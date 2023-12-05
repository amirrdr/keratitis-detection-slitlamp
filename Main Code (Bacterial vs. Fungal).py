import glob
from PIL import Image
import numpy as np
import tensorflow as tf
from keras.layers import *
from keras import Model
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import cv2
from scipy.ndimage import median_filter
import imageio
from sklearn.utils import shuffle

shape = (300, 300, 3)
TS = 0.1
EP = 150
BS = 15
CW = {0: 1, 1: 1.5}
MERGE_RATIO = 0.2
MED_KERNEL_SIZE = (10, 10)

FG_dir = './Data/Fungal/'
FG_names = ['Aspergillus', 'Candida', 'Fusarium', 'Others']
FG_data = []
for i in range(len(FG_names)):
    filelist = glob.glob(FG_dir + FG_names[i] + '/*.JPG')
    FG_data.append([np.array(Image.open(fname).resize((shape[1], shape[0]))) for fname in filelist])
FG_data = np.array(FG_data)
FG_data = np.concatenate((FG_data[0], FG_data[1], FG_data[2], FG_data[3]), axis=0)
FG_out = np.ones((len(FG_data)))

BC_dir = './Data/Bacterial/'
filelist = glob.glob(BC_dir + '*.JPG')
BC_data = [np.array(Image.open(fname).resize((shape[1], shape[0]))) for fname in filelist]
BC_data = np.array(BC_data)
BC_out = np.zeros((len(BC_data)))

FG_data, FG_out = shuffle(FG_data, FG_out)
BC_data, BC_out = shuffle(BC_data, BC_out)

inp = Input(shape=shape)

x_1 = Conv2D(10, 10, strides=2, activation='relu')(inp)
x = Dropout(0.1)(x_1)
x = BatchNormalization()(x)

x = Conv2D(20, 10, strides=2, activation='relu')(x)
x = Dropout(0.1)(x)
x = BatchNormalization()(x)

x = Conv2D(30, 10, strides=2, activation='relu')(x)
x = Dropout(0.1)(x)
x = BatchNormalization()(x)

x = GlobalAvgPool2D()(x)
x = Dense(15, activation='relu')(x)
x = Dropout(0.1)(x)
x = Dense(5, activation='relu')(x)
x = Dropout(0.1)(x)
out = Dense(1, activation='sigmoid')(x)

model = Model(inp, out)
heatmap_model = Model(inp, x_1)
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics='accuracy')
model.summary()

for i in range(4):
    X = np.concatenate((BC_data[(i * 1000):((i + 1) * 1000)], FG_data[(i * 500):((i + 1) * 500)]), axis=0)
    y = np.concatenate((BC_out[(i * 1000):((i + 1) * 1000)], FG_out[(i * 500):((i + 1) * 500)]), axis=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TS, shuffle=True)
    model.fit(X_train, y_train, validation_split=TS / 4, epochs=EP, batch_size=BS, class_weight=CW, shuffle=True)
    model.save('./Models/Main Model V2.1')
    results = model.evaluate(X_test, y_test)
    print('Evaluation Results in Phase (' + str(i + 1) + '):', results)
    y_pred = model.predict(X_test)
    y_pred[y_pred < 0.5] = 0
    y_pred[y_pred >= 0.5] = 1
    print('Confusion Matrix in Phase (' + str(i + 1) + '):')
    print(confusion_matrix(y_test, y_pred, normalize='true'))
    preds = heatmap_model(X_test)
    for j in range(len(preds)):
        tmp_1 = np.mean(preds[j], axis=2, dtype='float')
        tmp_1 = median_filter(tmp_1, size=MED_KERNEL_SIZE)
        tmp_1 = cv2.morphologyEx(tmp_1, cv2.MORPH_CLOSE, MED_KERNEL_SIZE)
        tmp_1 = cv2.applyColorMap(np.array(255 * tmp_1).astype('uint8'), cv2.COLORMAP_JET)
        imageio.imwrite('./Outputs/Main Code V2.1/Kernel/' + str(i + 1) + '/Image (' + str(j + 1) + ').jpg', tmp_1)
        imageio.imwrite('./Outputs/Main Code V2.1/Original/' + str(i + 1) + '/Image (' + str(j + 1) + ').jpg',
                        X_test[j])
        tmp = Image.fromarray(X_test[j]).resize((tmp_1.shape[1], tmp_1.shape[0]))
        tmp = (tmp - np.min(tmp)) / np.ptp(tmp)
        tmp = np.array(255 * tmp).astype('uint8')
        mrg_img = cv2.addWeighted(tmp_1, MERGE_RATIO, tmp, 1 - MERGE_RATIO, 0)
        imageio.imwrite('./Outputs/Main Code V2.1/Merged/' + str(i + 1) + '/Image (' + str(j + 1) + ').jpg', mrg_img)
heatmap_model.save('./Models/Heatmap Model V2.1')
