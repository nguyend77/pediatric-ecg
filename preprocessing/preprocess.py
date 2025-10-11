import numpy as np
import pandas as pd
import wfdb
from pyts.image import MarkovTransitionField
from tqdm import tqdm

df = pd.read_csv('ecg_data.csv')
# path to Child_ecg folder
path = 'data/Child_ecg/'
# image size for MTF
size = 256

def retrieve_mtf(address):
    # use float32 to save 50% memmory
    arr = np.zeros((size, size, 12), dtype=np.float32)
    # crop to first 4096 sampling points (2^12)
    ecg = wfdb.rdsamp(path+address)[0][:4096, :]
    for lead in range(12):
        # create a row vector from a 1D array and fit data to MTF, downscale data to 256 points
        mtf = MarkovTransitionField(image_size=size, strategy='uniform').fit_transform(ecg[:, lead].reshape(1, -1))
        # output of fit_transform is (1, 256, 256), append first element to arr
        arr[:, :, lead] = mtf[0]
    return arr

# save MTFs as X
filename = df['Filename'].to_list()
length = len(filename)
X = np.memmap('input/X.npy', dtype=np.float32, mode='w+', shape=(length, size, size, 12))
for i in tqdm(range(length)):
    X[i] = retrieve_mtf(filename[i])
X.flush()

# save diagnosis as Y
np.save('input/Y.npy', df['diagnosis'])