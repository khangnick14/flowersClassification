# %%
import tensorflow as tf
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from tensorflow import keras
from keras.models import Sequential
from sklearn.model_selection import train_test_split


# %%
data_dir = 'Flowers'

# %%
labels = ['Babi', 'Calimerio', 'Chrysanthemum',
          'Hydrangeas', 'Lisianthus', 'Pingpong', 'Rosy', 'Tana']


def prepare_data(data_dir):
    data = []
    for img_class in os.listdir(data_dir):
        class_path = os.path.join(data_dir, img_class)
        img_index = labels.index(img_class)
        for img in os.listdir(class_path):
            try:
                img_path = os.path.join(data_dir, img_class, img)
                img_arr = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img_arr = cv2.resize(img_arr, (255, 255))
                data.append([img_arr, img_index])
            except Exception as e:
                print(e)
    return np.array(data, dtype=object)


data = prepare_data(data_dir)

# %%
# Separate label and array
l = []
for i in data:
    l.append(labels[i[1]])
pd.Series(l).value_counts(sort=False).plot(kind='bar')

# %%
x = []
y = []
for img, label in data:
    x.append(img)
    y.append(label)

# %%
babi_list = len([i for i in y if i == 0])
calimerio_list = len([i for i in y if i == 1])
chrysanthemm_list = len([i for i in y if i == 2])
hydrangeas_list = len([i for i in y if i == 3])
lisianths_list = len([i for i in y if i == 4])
pingpong_list = len([i for i in y if i == 5])
rosy_list = len([i for i in y if i == 6])
tana_list = len([i for i in y if i == 7])

flower_list = [babi_list, calimerio_list, chrysanthemm_list,
               hydrangeas_list, lisianths_list, pingpong_list, rosy_list, tana_list]
plt.figure(figsize=(15, 15))
plt.pie(flower_list, labels=labels, startangle=90, colors=[
        'red', 'green', 'blue', 'yellow', 'purple', 'pink', 'orange', 'brown'], autopct='%1.1f%%')
plt.legend()
plt.show()
# Normalize data
# %%
# Preprocessing data
# %%
# Build model
