import numpy as np
import pandas as pd
import tensorflow as tf
import wfdb
from pyts.image import MarkovTransitionField

df = pd.read_csv('ecg_data.csv')
# path to Child_ecg folder
path = 'data/Child_ecg/'
# image size for MTF
size = 256

def retrieve_mtf(address):
    # use float32 to save 50% memmory
    arr = np.zeros((size, size, 12), dtype=np.float32)
    # crop to get uniform inputs
    ecg = wfdb.rdsamp(path+address)[0][:5120, :]
    for lead in range(12):
        # create a row vector from a 1D array and fit data to MTF, downscale data to 256 points
        mtf = MarkovTransitionField(image_size=size, strategy='uniform').fit_transform(ecg[:, lead].reshape(1, -1))
        # output of fit_transform is (1, 256, 256), append first element to arr
        arr[:, :, lead] = mtf[0]
    return arr

def generator():
    for index, row in df.iterrows():
        yield retrieve_mtf(row['Filename']), row['diagnosis']

signature = (tf.TensorSpec(shape=(size, size, 12), dtype=tf.float32), tf.TensorSpec(shape=(), dtype=tf.float32))

dataset = tf.data.Dataset.from_generator(generator, output_signature=signature)