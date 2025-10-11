import numpy as np
import pandas as pd
import wfdb
from pyts.image import MarkovTransitionField

# pull data from GitHub repo
df = pd.read_csv('https://raw.githubusercontent.com/nguyend77/pediatric-ecg/refs/heads/main/ecg_data.csv')

# path to Child_ecg folder
path = 'data/Child_ecg/'

def retrieve_mtf(address):
    arr = np.zeros((1, 5000, 5000, 12), dtype=np.float64)
    # crop to first 5000 sampling points (10s)
    ecg = wfdb.rdsamp(path+address)[0][:5000, :]
    for lead in range(12):
        # create a row vector from a 1D array and fit data to MTF
        mtf = MarkovTransitionField(strategy='uniform').fit_transform(ecg[:, lead].reshape(1, -1))
        # output of fit_transform is (1, 5000, 5000), append first element to arr
        arr[0, :, :, lead] = mtf[0]
    return arr