# %%
import tensorflow as tf
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tensorflow import keras
from keras.models import Sequential
from sklearn.model_selection import train_test_split


# %%
data_dir = 'Flowers'

# %%
labels = os.listdir(data_dir)


def prepare_data(data_dir):
    data = []
    for img_class in os.listdir(data_dir):
        for img in os.listdir(os.path.join(data_dir, img_class)):
            try:
                img_path = os.path.join(data_dir, img_class, img)
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (255, 255))
                data.append([img, img_class])
            except Exception as e:
                print(e)
    return np.array(data)


data = prepare_data(data_dir)

# %%
l = []
for i in data:
    l.append(labels[i[1]])
sns.set_style('dark')
sns.countplot(l)


# %%
# Scale data
data = data.map(lambda x, y: (x, y/255))
# %%
x = []
y = []
for img, label in data:
    x.append(img)
    y.append(label)

# %%
# Preprocessing data
# %%
# Build model
