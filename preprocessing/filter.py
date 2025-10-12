import numpy as np
import pandas as pd

df = pd.read_csv('data/AttributesDictionary.csv', usecols=['Filename', 'Sampling_point', 'Lead', 'ICD-10 code'])
# regular expression (regex) pattern
cardiovascular_disease = 'I40.0|I40.9|I51.4|I42.0|I42.2|I42.9|Q24.8|M30.3|Q21.0|Q21.1|Q21.2|Q21.3|Q22.1|Q25.0|Q25.6|I37.0'
df['diagnosis'] = df['ICD-10 code'].str.contains(cardiovascular_disease).astype(int)
# limit to 12 lead ECGs
df = df[df['Lead'] == 12]
# drop records of less than threshold (~10s)
df = df[df['Sampling_point'] >= 5120]
df = df.drop(columns=['Sampling_point', 'Lead', 'ICD-10 code'])
print(df['diagnosis'].value_counts())
df.to_csv('ecg_data.csv', index=False)