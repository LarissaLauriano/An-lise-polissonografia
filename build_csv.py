import os, glob
import numpy as np
import pandas as pd

# Define root data paths, where the NSRR data is stored
root_dir = 'Datasets/NSRR/'
assert os.path.isdir(root_dir)

# Define the output path
out_dir = 'output/'
if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

'''
CCSHS
https://sleepdata.org/datasets/ccshs
'''
# Load demographics CSV file
df_ccshs = pd.read_csv(
    root_dir + 'ccshs/datasets/ccshs-trec-dataset-0.6.0.csv', 
    usecols=['nsrrid', 'age', 'ethnicity', 'male', 'bmi', 'race3', 'ahi_a0h3', 'yinsdx', 'yhtn', 'ydiab', 'ydep']
)

# Rename 
df_ccshs['race3'].replace({1: 'caucasian', 2: 'african', 3: 'other'}, 
                          inplace=True)
df_ccshs.loc[df_ccshs['ethnicity'] == 1, 'race3'] = 'hispanic'
df_ccshs.drop(columns=['ethnicity'], inplace=True)
df_ccshs.rename(columns={
    'race3': 'ethnicity', 
    'nsrrid': 'subj', 
    'ahi_a0h3': 'ahi',
    'yinsdx': 'insomnia',
    'yhtn': 'hypertension',
    'ydiab': 'diabete',
    'ydep': 'depression'}, inplace=True)

# Remove subject for which we don't have the data
df_ccshs = df_ccshs[~df_ccshs['subj'].isin([1800639, 1800842])]

# Convert to str
df_ccshs['subj'] = df_ccshs['subj'].astype(str)
df_ccshs.set_index('subj', inplace=True)

# Export demographics to CSV file
df_ccshs['dataset'] = 'CCSHS'
df_ccshs.to_csv(out_dir + "demo_nsrr_ccshs.csv")

print(df_ccshs.shape[0], "unique nights.")
print(df_ccshs['set'].value_counts())
df_ccshs.head()

'''
CFS
https://sleepdata.org/datasets/cfs
'''
df_cfs = pd.read_csv(
    root_dir + 'cfs/datasets/cfs-visit5-dataset-0.4.0.csv', 
    usecols=['nsrrid', 'age', 'SEX', 'race', 'ethnicity', 'bmi', 'ahi_a0h3',
             'AbnorEEG', 'AbnorEye', 'QuEEG1', 'QuHR', 'QuEOGL', 'QuChin', 
             'RemNRemPr', 'Stg1Stg2Pr', 'Stg2Stg3Pr', 'WakSlePr', 
             'INSODIAG', 'htndx', 'DIADIAG', 'DIANARC', 'DEPDIAG'])

# Rename for ease-of-use
df_cfs.rename(columns={
    'nsrrid': 'subj',
    'SEX': 'male',
    'ahi_a0h3': 'ahi',
    'INSODIAG': 'insomnia',
    'htndx': 'hypertension',
    'DIADIAG': 'diabete',
    'DIANARC': 'narcolepsy',
    'DEPDIAG': 'depression'
}, inplace=True)

df_cfs['race'].replace({1: 'caucasian', 2: 'african', 3: 'other'}, inplace=True)
df_cfs.loc[df_cfs['ethnicity'] == 1, 'race'] = 'hispanic'
df_cfs.drop(columns=['ethnicity'], inplace=True)
df_cfs.rename(columns={'race': 'ethnicity'}, inplace=True)

# Remove subjects with no EEG / hypnogram file
df_cfs = df_cfs[~df_cfs['subj'].isin([800269, 802230, 801601, 801811, 801841, 801953, 802234])]

# Remove Ss with bad data quality or unreliable scoring (DISABLED)
# df_cfs = df_cfs[df_cfs['RemNRemPr'] != 1]
# df_cfs = df_cfs[df_cfs['Stg1Stg2Pr'] != 1]
# df_cfs = df_cfs[df_cfs['Stg2Stg3Pr'] != 1]
# df_cfs = df_cfs[df_cfs['WakSlePr'] != 1]
# df_cfs = df_cfs[df_cfs['AbnorEEG'] != 1]
# df_cfs = df_cfs[df_cfs['AbnorEye'] != 1]
# df_cfs = df_cfs[df_cfs['QuEEG1'] >= 4] # >= 4 = signal good for > 75% of the time
# df_cfs = df_cfs[df_cfs['QuEOGL'] >= 4]
# df_cfs = df_cfs[df_cfs['QuChin'] >= 4]

# Drop columns
df_cfs.drop(
    columns=['RemNRemPr', 'AbnorEEG', 'AbnorEye', 'QuEEG1', 'QuHR', 'QuEOGL', 
             'QuChin', 'RemNRemPr', 'Stg1Stg2Pr', 'Stg2Stg3Pr', 'WakSlePr'], 
    inplace=True)

# Convert to str
df_cfs['subj'] = df_cfs['subj'].astype(str)
df_cfs.set_index('subj', inplace=True)

# Export demographics to CSV file
df_cfs['dataset'] = 'CFS'
df_cfs.to_csv(out_dir + "demo_nsrr_cfs.csv")

print(df_cfs.shape[0], "unique nights.")
print(df_cfs['set'].value_counts())
df_cfs.head()

'''
CHAT
https://sleepdata.org/datasets/chat
'''
usecols = ['nsrrid', 'male', 'ageyear_at_meas', 'overall', 
           'ant5', 'chi3', 'race3', 'ahi_a0h3', 'med1o2', 'med1m2', 'med1i2', 'med1h2']

df_chat = pd.read_csv(
    root_dir + 'chat/datasets/chat-baseline-dataset-0.11.0.csv', usecols=usecols)

# Rename columns
df_chat.rename(columns={'nsrrid': 'subj',
                        'ageyear_at_meas': 'age',
                        'ant5': 'bmi',
                        'chi3': 'ethnicity',
                        'race3': 'race',
                        'ahi_a0h3': 'ahi',
                        'med1o2': 'insomnia',
                        'med1m2': 'hypertension',
                        'med1i2': 'diabete',
                        'med1h2': 'depression'
                      }, inplace=True)

df_chat['race'].replace({1: 'caucasian', 2: 'african', 3: 'other'}, inplace=True)
df_chat.loc[df_chat['ethnicity'] == 1, 'race'] = 'hispanic'
df_chat.drop(columns=['ethnicity'], inplace=True)
df_chat.rename(columns={'race': 'ethnicity'}, inplace=True)

# Keep only "Excellent" quality study (DISABLED)
# print(df_chat[df_chat['overall'] < 6].shape[0], 'subjects with bad PSG data quality will be removed.')
# df_chat = df_chat[df_chat['overall'] >= 6]

# Remove nights for which sex is NaN
df_chat.dropna(subset=['male'], inplace=True)

# Convert to str
df_chat['subj'] = df_chat['subj'].apply(lambda x: str(x).zfill(4))
df_chat.set_index('subj', inplace=True)
df_chat['male'] = df_chat['male'].astype(int)
df_chat['age'] = df_chat['age'].astype(float)

# Define training / testing
# Keep only a random subset of ~350 subjects for training (leave 100 for testing)
df_chat["set"] = "excluded"
idx_train = df_chat.sample(n=353, replace=False, random_state=42).index
idx_test = np.setdiff1d(df_chat.index, idx_train)
# Now we keep 100 random participants of ``idx_test`` for testing
rs = np.random.RandomState(42)
idx_test = rs.choice(idx_test, size=100, replace=False)
df_chat.loc[idx_train, "set"] = "training"
df_chat.loc[idx_test, "set"] = "testing"

# Export demographics to CSV file
df_chat['dataset'] = 'CHAT'
# df_chat.to_csv(out_dir + "demo_nsrr_chat.csv")

print(df_chat.shape[0], "unique nights.")
print(df_chat['set'].value_counts())
df_chat.head()

'''
HomePAP
https://sleepdata.org/datasets/homepap
'''
desc_dir = root_dir + 'homepap/datasets/homepap-baseline-dataset-0.1.0.csv'
hypno_dir = root_dir + 'homepap/polysomnography/annotations-events-profusion/lab/full/'
usecols = ['nsrrid', 'treatmentarm', 'age', 'gender', 'bmi', 'ahi_full', 'race3', 'ethnicity',
           'dxhtn', 'dxdiab', 'dxdep']

df_hpap = pd.read_csv(desc_dir, usecols=usecols)

df_hpap.rename(columns={
    'nsrrid': 'subj',
    'race3': 'race',
    'ahi_full': 'ahi',
    'dxhtn': 'hypertension',
    'dxdiab': 'diabete',
    'dxdep': 'depression'
    }, inplace=True)

df_hpap['race'].replace({1: 'caucasian', 2: 'african', 3: 'other'}, inplace=True)
df_hpap.loc[df_hpap['ethnicity'] == 1, 'race'] = 'hispanic'
df_hpap.drop(columns=['ethnicity'], inplace=True)
df_hpap.rename(columns={'race': 'ethnicity'}, inplace=True)

df_hpap['male'] = df_hpap['gender'].replace({'M': 1, 'F': 0})

# Drop NaN
# df_hpap.dropna(inplace=True)

# Keep only in-lab PSG
df_hpap = df_hpap[df_hpap['treatmentarm'] == 1]
df_hpap.drop(columns=['treatmentarm', 'gender'], inplace=True)

# Keep only full night
files = [int(f.split('-')[3]) for f in os.listdir(hypno_dir)]
good = np.intersect1d(files, df_hpap['subj'].values, assume_unique=True)
df_hpap = df_hpap[df_hpap['subj'].isin(good)]

# Convert to str
df_hpap['subj'] = df_hpap['subj'].astype(str)
df_hpap.set_index('subj', inplace=True)

# Because of the few number of participants, we only use HomePaP as a training dataset
df_hpap["set"] = "training"

# Export demographics to CSV file
df_hpap['dataset'] = 'HOMEPAP'
df_hpap.to_csv(out_dir + "demo_nsrr_homepap.csv")

print(df_hpap.shape[0], 'subjects remaining')
print(df_hpap['set'].value_counts())
df_hpap.head(10)

'''
MESA
https://sleepdata.org/datasets/mesa
'''
# BioLinCC
df_mesa_bio = pd.read_csv("data/mesae5_drepos_20151101.csv", usecols=['mesaid', 'bmi5c', 'htn5c', 'dm035c'])

# Diabetes: 0=normal, 1=impaired, 2=treated diabete, 3=untreated diabete
df_mesa_bio.rename(
    columns={'mesaid': 'subj', 'bmi5c': 'bmi', 'htn5c': 'hypertension', 'dm035c': 'diabete'}, inplace=True)

df_mesa_bio
desc_dir = root_dir + 'mesa/datasets/mesa-sleep-dataset-0.3.0.csv'

# No BMI for MESA -- see BioLinCC instead
df_mesa = pd.read_csv(
    desc_dir, usecols=['mesaid', 'gender1', 'sleepage5c', 'overall5', 'race1c', 'ahi_a0h3', 'insmnia5'])

# Rename columns
df_mesa.rename(columns={'mesaid': 'subj',
                        'gender1': 'gender',
                        'sleepage5c': 'age',
                        'overall5': 'overall',
                        'race1c': 'ethnicity',
                        'ahi_a0h3': 'ahi',
                        'insmnia5': 'insomnia',
                      }, inplace=True)

df_mesa['ethnicity'].replace({1: 'caucasian', 2: 'other', 3: 'african', 4: 'hispanic'}, inplace=True)

# Keep only "Excellent" quality study (DISABLED)
# print(df_mesa[df_mesa['overall'] < 6].shape[0], 
#       'subjects with bad PSG data quality will be removed.')
# df_mesa = df_mesa[df_mesa['overall'] >= 6]
df_mesa['male'] = (df_mesa['gender'] == 1).astype(int)
df_mesa.drop(columns=['gender'], inplace=True)

# Merge with BioLinCC
df_mesa = df_mesa.merge(df_mesa_bio, how="left")

# Convert to str
df_mesa['subj'] = df_mesa['subj'].apply(lambda x: str(x).zfill(4))
df_mesa.set_index('subj', inplace=True)

# In MESA, there are a large number of EDF files that are missing
# We therefore only keep subjects that have valid data here.
edf_files = sorted(glob.glob(root_dir + 'mesa/polysomnography/edfs/*.edf'))
edf_files = [os.path.basename(c)[-8:-4] for c in edf_files]
df_mesa = df_mesa.loc[edf_files] 

# Export demographics to CSV file
df_mesa['dataset'] = 'MESA'
df_mesa.to_csv(out_dir + "demo_nsrr_mesa.csv")

print(df_mesa.shape[0], 'subjects remaining')
print(df_mesa['set'].value_counts())
df_mesa.head(10)

'''
MrOS
https://sleepdata.org/datasets/mros
'''
desc_dir = root_dir + 'mros/datasets/mros-visit1-dataset-0.4.0.csv'
usecols = ['nsrrid', 'visit', 'gender', 'vsage1', 'pooveral', 'hwbmi', 'gierace', 'poordi3', 
           'slinsomn', 'mhbp', 'mhdiab', 'slnarc']

df_mros = pd.read_csv(desc_dir, usecols=usecols, low_memory=False)

# Rename columns
df_mros.rename(columns={'nsrrid': 'subj',
                        'vsage1': 'age',
                        'pooveral': 'overall',
                        'hwbmi': 'bmi',
                        'gierace': 'ethnicity',
                        'poordi3': 'ahi',
                        'slinsomn': 'insomnia',
                        'mhbp': 'hypertension',
                        'mhdiab': 'diabete',
                        'slnarc': 'narcolepsy',
                      }, inplace=True)

# Keep only "Excellent" quality study
# print(df_mros[df_mros['overall'] < 6].shape[0], 
#       'subjects with bad PSG data quality will be removed.')
# df_mros = df_mros[df_mros['overall'] >= 6]

# They should all be male in MrOS!
df_mros['male'] = (df_mros['gender'] == 2).astype(int)
df_mros['ethnicity'].replace({1: 'caucasian', 2: 'african', 3: 'other', 
                              4: 'hispanic', 5: 'other'}, inplace=True)

# Keep only first visit
df_mros = df_mros[df_mros['visit'] == 1]

# Convert to str
df_mros['subj'] = df_mros['subj'].apply(lambda x: str(x).zfill(4))
df_mros.set_index('subj', inplace=True)

# Fix an invalid 'A' in BMI
df_mros.loc[df_mros['bmi'] == 'A', 'bmi'] = np.nan
df_mros['bmi'] = df_mros['bmi'].astype(float)

# Export demographics to CSV file
df_mros['dataset'] = 'MROS'
df_mros.to_csv(out_dir + "demo_nsrr_mros.csv")

print(df_mros.shape[0], 'subjects remaining')
print(df_mros['set'].value_counts())
df_mros.head(10)

'''
SHHS
https://sleepdata.org/datasets/shhs
'''
desc_dir = root_dir + 'shhs/datasets/shhs1-dataset-0.16.0.csv'
usecols = ['nsrrid', 'visitnumber', 'gender', 'age_s1', 'overall_shhs1', 
           'race', 'bmi_s1', 'ahi_a0h3', 'ethnicity', 'HTNDerv_s1', 'ParRptDiab']

df_shhs = pd.read_csv(desc_dir, usecols=usecols)

# Rename columns
df_shhs.rename(columns={'nsrrid': 'subj',
                        'gender1': 'gender',
                        'age_s1': 'age',
                        'overall_shhs1': 'overall',
                        'bmi_s1': 'bmi',
                        'ahi_a0h3': 'ahi',
                        'HTNDerv_s1': 'hypertension',
                        'ParRptDiab': 'diabete'
                      }, inplace=True)

df_shhs['race'].replace({1: 'caucasian', 2: 'african', 3: 'other'}, inplace=True)
df_shhs.loc[df_shhs['ethnicity'] == 1, 'race'] = 'hispanic'
df_shhs.drop(columns=['ethnicity'], inplace=True)
df_shhs.rename(columns={'race': 'ethnicity'}, inplace=True)

# Keep only "Excellent" quality study
# print(df_shhs[df_shhs['overall'] < 6].shape[0], 
#       'subjects with bad PSG data quality will be removed.')
# df_shhs = df_shhs[df_shhs['overall'] >= 6]

df_shhs['male'] = (df_shhs['gender'] == 1).astype(int)

# Keep only first visit
df_shhs = df_shhs[df_shhs['visitnumber'] == 1]

# Convert to str
df_shhs['subj'] = df_shhs['subj'].apply(lambda x: str(x).zfill(4))
df_shhs.set_index('subj', inplace=True)

# Export demographics to CSV file
df_shhs['dataset'] = 'SHHS'
df_shhs.to_csv(out_dir + "demo_nsrr_shhs.csv")

print(df_shhs.shape[0], 'subjects remaining')
print(df_shhs['set'].value_counts())
df_shhs.head(10)
