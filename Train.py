import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import itertools
from glob import glob
from PIL import Image
from sklearn.metrics import confusion_matrix
from  sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

base_crack_Negative = "/Users/vodinhnguyen/Downloads/Python/CrackCNN/Input/NonCracks"
base_crack_Positive = "/Users/vodinhnguyen/Downloads/Python/CrackCNN/Input/Cracks"

crack_Negative_df = pd.DataFrame(columns=('path', 'image', 'label'),index=np.arange(0,200))
crack_Positive_df = pd.DataFrame(columns=('path', 'image', 'label'),index=np.arange(0,200))

imageid_path_dict_Negative = np.array([x for x in glob(os.path.join(base_crack_Negative, '*.jpg'))])
imageid_path_dict_Positive = np.array([x for x in glob(os.path.join(base_crack_Positive, '*.jpg'))])

crack_Negative_df['path'] = imageid_path_dict_Negative
crack_Negative_df['label'] = 0
crack_Negative_df['image'] = crack_Negative_df['path'].map(lambda x: np.asarray(Image.open(x).resize((75,100))))

crack_Positive_df['path']=imageid_path_dict_Positive
crack_Positive_df['label']=1
crack_Positive_df['image']=crack_Positive_df['path'].map(lambda x: np.asarray(Image.open(x).resize((75,100))))

print("Negative:",crack_Negative_df["image"][0].shape)
print("Positive:",crack_Positive_df["image"][0].shape)

print(imageid_path_dict_Negative.shape,imageid_path_dict_Positive.shape)

crack_df = crack_Negative_df.append(crack_Positive_df)  # 将两个dataframe合并
crack_df.reset_index(drop=True,inplace=True)  # 重置检索
print(crack_df.shape)
features=crack_df.drop(columns=['label'],axis=1)
target=crack_df["label"]

x_train_o, x_test_o, y_train, y_test = train_test_split(features, target, test_size=0.20,random_state=1234)

x_train = np.asarray(x_train_o['image'].tolist())
x_test = np.asarray(x_test_o['image'].tolist())

x_train_mean = np.mean(x_train)
x_train_std = np.std(x_train)

x_test_mean = np.mean(x_test)
x_test_std = np.std(x_test)

x_train = (x_train - x_train_mean)/x_train_std
x_test = (x_test - x_test_mean)/x_test_std

# Reshape image in 3 dimensions (height = 100px, width = 100px , canal = 3)
x_train = x_train.reshape(x_train.shape[0], *(75, 100, 3))
x_test = x_test.reshape(x_test.shape[0], *(75, 100, 3))
input_shape = (75, 100, 3)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',padding = 'Same',input_shape=input_shape))
model.add(Conv2D(32,kernel_size=(3, 3), activation='relu',padding = 'Same',))
model.add(MaxPool2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu',padding = 'Same'))
model.add(Conv2D(64, (3, 3), activation='relu',padding = 'Same'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.40))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model.fit(x_train, y_train, validation_split=0.1,epochs=50, batch_size=256)

history = model.history
print(history.history)

fig, ax = plt.subplots()

ax.plot(history.history['accuracy'], label='training accuracy')
ax.plot(history.history['val_accuracy'], label='val accuracy')

ax.set_title('model accuracy', {'size': 16})
ax.set_xlabel('epoch', {'size': 16})
ax.set_ylabel('accuracy', {'size': 16})

plt.tick_params(labelsize=14)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

fig, ax = plt.subplots()

ax.plot(history.history['loss'], label='training loss')
ax.plot(history.history['val_loss'], label='val loss')

ax.set_title('model loss', {'size': 16})
ax.set_xlabel('epoch', {'size': 16})
ax.set_ylabel('loss', {'size': 16})

plt.tick_params(labelsize=14)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score,recall_score, f1_score,roc_auc_score

y_score = model.predict_classes(x_test)
y_score_1 = model.predict(x_test)

import seaborn as sns
cm = confusion_matrix(y_test, y_score)
print('confusion_matrix\n', cm)

print('accuracy:{}'.format(accuracy_score(y_test, y_score)))
print('precision:{}'.format(precision_score(y_test, y_score)))
print('recall:{}'.format(recall_score(y_test, y_score)))
print('f1-score:{}'.format(f1_score(y_test, y_score)))

f,ax=plt.subplots()
sns.heatmap(cm,annot=True,ax=ax,fmt='.4g') #Draw a heat map

ax.set_title('confusion matrix',fontsize=16) #title
ax.set_xlabel('Predict',fontsize=16) #Xaxis
ax.set_ylabel('True',fontsize=16) #Yaxis
fig, ax = plt.subplots(figsize=(5, 4.4))
#AUC valve
auc = roc_auc_score(y_test,y_score_1)
#Draw curves
fpr, tpr, thresholds = roc_curve(y_test,y_score_1)
ax.plot(fpr, tpr, linewidth = 2,label='AUC=%.4f' % auc)
ax.plot([0,1],[0,1], 'k--')


ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel('False Postivie Rate',{'size':16})
ax.set_ylabel('True Positive Rate',{'size':16})

ax.plot([0,1],[0,1], 'k--',lw=2)

plt.tick_params(labelsize=14)

plt.legend()
plt.tight_layout()
plt.show()