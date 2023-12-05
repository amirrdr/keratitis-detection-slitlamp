import glob
from PIL import Image
import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from scipy.stats import sem
import scikitplot as skplt
import matplotlib.pyplot as plt

shape = (300, 300, 3)
TS = 0.2

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

AC_dir = './Data/Acanthamoeba/'
filelist = glob.glob(AC_dir + '*.JPG')
AC_data = [np.array(Image.open(fname).resize((shape[1], shape[0]))) for fname in filelist]
AC_data = np.array(AC_data)
AC_out = 2 * np.ones((len(AC_data)))

print('Number of Acanthamoeba Keratitis Samples:', len(AC_data))

X = np.concatenate((BC_data, FG_data, AC_data), axis=0)
y = np.concatenate((BC_out, FG_out, AC_out), axis=0)

_, X_test, _, y_test = train_test_split(X, y, test_size=TS, shuffle=True)

model = tf.keras.models.load_model('./Models/Main Model V2.2')

preds = model.predict(X_test)
print('Keratitis vs. Healthy:')
print('AUC Value:', roc_auc_score(y_test, preds))
fpr, tpr, thresholds = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='example estimator')
display.plot()
plt.savefig('./Figures/ROC (Keratitis vs. Healthy).JPG')
plt.show()

n_bootstraps = 1000
rng_seed = 42
bootstrapped_scores = []

rng = np.random.RandomState(rng_seed)
for i in range(n_bootstraps):
    indices = rng.randint(0, len(preds), len(preds))
    if len(np.unique(y_test[indices])) < 2:
        continue

    score = roc_auc_score(y_test[indices], preds[indices])
    bootstrapped_scores.append(score)
    print("Bootstrap #{} ROC area: {:0.3f}".format(i + 1, score))

sorted_scores = np.array(bootstrapped_scores)
sorted_scores.sort()

confidence_lower = sorted_scores[int(0.05 * len(sorted_scores))]
confidence_upper = sorted_scores[int(0.95 * len(sorted_scores))]
print("Confidence Interval for the Score: [{:0.3f} - {:0.3}]".format(confidence_lower, confidence_upper))


shape = (300, 300, 3)
TS = 0.1

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

X = np.concatenate((BC_data, FG_data), axis=0)
y = np.concatenate((BC_out, FG_out), axis=0)
_, X_test, _, y_test = train_test_split(X, y, test_size=TS, shuffle=True)

model = tf.keras.models.load_model('./Models/Main Model V2.1')

preds = model.predict(X_test)
print('Bacterial Keratitis vs. Fungal Keratitis:')
print('AUC Value:', roc_auc_score(y_test, preds))
fpr, tpr, thresholds = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='example estimator')
display.plot()
plt.savefig('./Figures/ROC (Bacterial Keratitis vs. Fungal Keratitis).JPG')
plt.show()

n_bootstraps = 1000
rng_seed = 42
bootstrapped_scores = []

rng = np.random.RandomState(rng_seed)
for i in range(n_bootstraps):
    indices = rng.randint(0, len(preds), len(preds))
    if len(np.unique(y_test[indices])) < 2:
        continue

    score = roc_auc_score(y_test[indices], preds[indices])
    bootstrapped_scores.append(score)
    print("Bootstrap #{} ROC area: {:0.3f}".format(i + 1, score))

sorted_scores = np.array(bootstrapped_scores)
sorted_scores.sort()

confidence_lower = sorted_scores[int(0.05 * len(sorted_scores))]
confidence_upper = sorted_scores[int(0.95 * len(sorted_scores))]
print("Confidence Interval for the Score: [{:0.3f} - {:0.3}]".format(confidence_lower, confidence_upper))

shape = (100, 100, 3)
TS = 0.1

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

X = np.concatenate((FA_data, FG_data), axis=0)
y = np.concatenate((FA_out, FG_out), axis=0)

_, X_test, _, y_test = train_test_split(X, y, test_size=TS, shuffle=True)

model = tf.keras.models.load_model('./Models/Main Model V3.0')

preds = model.predict(X_test)
print('Candida or Others Types of Keratitis vs. Aspergillus or Fusarium Keratitis:')
print('AUC Value:', roc_auc_score(y_test, preds))
fpr, tpr, thresholds = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='example estimator')
display.plot()
plt.savefig('./Figures/ROC (Candida or Others Types of Keratitis vs. Aspergillus or Fusarium Keratitis).JPG')
plt.show()

n_bootstraps = 1000
rng_seed = 42
bootstrapped_scores = []

rng = np.random.RandomState(rng_seed)
for i in range(n_bootstraps):
    indices = rng.randint(0, len(preds), len(preds))
    if len(np.unique(y_test[indices])) < 2:
        continue

    score = roc_auc_score(y_test[indices], preds[indices])
    bootstrapped_scores.append(score)
    print("Bootstrap #{} ROC area: {:0.3f}".format(i + 1, score))

sorted_scores = np.array(bootstrapped_scores)
sorted_scores.sort()

confidence_lower = sorted_scores[int(0.05 * len(sorted_scores))]
confidence_upper = sorted_scores[int(0.95 * len(sorted_scores))]
print("Confidence Interval for the Score: [{:0.3f} - {:0.3}]".format(confidence_lower, confidence_upper))
