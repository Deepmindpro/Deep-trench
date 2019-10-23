import pandas as pd
import os
import numpy as np
import itertools
import collections
import re
import nltk as nl
from nltk.corpus import stopwords

stopset = set(stopwords.words('english'))

stopset.add('.');
stopset.add(',');
stopset.add('--');
stopset.add(';');
stopset.add("'");
stopset.add("-");
stopset.add("...");
stopset.add(" ")

''' Load data '''
folder = 'C://Users//i776140//Python data files'
filename = 'agr_en_train.csv'
df = pd.read_csv(os.path.join(folder, filename), sep=';', header=None, usecols=[0, 1, 2])

df_NAG = df[df[2] == "NAG"]
df_OAG = df[df[2] == "OAG"]
df_CAG = df[df[2] == "CAG"]

'''Processing of NAG category starts'''
z = []
for i in df_NAG.iloc[:, 1]:
    if i is not np.nan:
        z.append(i.split(sep=' '))
    else:
        z.append(str(i))
z_flattened = list(itertools.chain.from_iterable(z))
# print(z_flattened)

# to remove special characters

regex = re.compile('a-zA-Z')
Lst_NAG = []
for i in z_flattened:
    Lst_NAG.append(regex.sub('', i))
Lst_NAG = [i for i in Lst_NAG if i not in stopset]  # Removing characters that appeared in Stopset
Freq_NAG = nl.FreqDist(Lst_NAG).most_common()
d_NAG = {}  # converting the list tuples into a dictionary
for i in range(0, len(Freq_NAG)):
    t = Freq_NAG[i]
    d_NAG.update({t[0]: t[1]})
# print('Dictionary for NAG:', d_NAG)

'''Processing of OAG category starts'''
z = []
for i in df_OAG.iloc[:, 1]:
    if i is not np.nan:
        z.append(i.split(sep=' '))
    else:
        z.append(str(i))
z_flattened = list(itertools.chain.from_iterable(z))
# to remove special characters
regex = re.compile('a-zA-Z')
Lst_OAG = []
for i in z_flattened:
    Lst_OAG.append(regex.sub('', i))
Lst_OAG = [i for i in Lst_OAG if i not in stopset]  # Removing characters that appeared in Stopset
Freq_OAG = nl.FreqDist(Lst_OAG).most_common()
d_OAG = {}  # converting the list tuples into a dictionary
for i in range(0, len(Freq_OAG)):
    t = Freq_OAG[i]
    d_OAG.update({t[0]: t[1]})
print('Dictionary for OAG:', d_OAG)

'''Processing of CAG category starts'''
z = []
for i in df_CAG.iloc[:, 1]:
    if i is not np.nan:
        z.append(i.split(sep=' '))
    else:
        z.append(str(i))
z_flattened = list(itertools.chain.from_iterable(z))
# to remove special characters
regex = re.compile('a-zA-Z')
Lst_CAG = []
for i in z_flattened:
    Lst_CAG.append(regex.sub('', i))
Lst_CAG = [i for i in Lst_CAG if i not in stopset]  # Removing characters that appeared in Stopset
Freq_CAG = nl.FreqDist(Lst_CAG).most_common()
d_CAG = {}  # converting the list tuples into a dictionary
for i in range(0, len(Freq_CAG)):
    t = Freq_CAG[i]
    d_CAG.update({t[0]: t[1]})
# print('Dictionary for CAG:', d_CAG)

str = "The BJP is friend of Pakistan Modi betrays India"
val_OAG = 0
val_NAG = 0
val_CAG = 0
d_str = []
d_str = str.split(" ")
print(d_str[0])
print(len(d_str))

'''ML Coding on OAG category starts'''
for i in range(0, len(d_str)):
    if d_OAG.get(d_str[i], 'NA') != 'NA':
        val_OAG = pd.to_numeric(d_OAG.get(d_str[i])) + val_OAG
    # val = val + pd.to_numeric(d_OAG[d_OAG.get(d_str[i])])
    else:
        continue
print('cumulate value OAG:', val_OAG)

'''ML Coding on NAG category starts'''
for i in range(0, len(d_str)):
    if d_NAG.get(d_str[i], 'NA') != 'NA':
        val_NAG = pd.to_numeric(d_NAG.get(d_str[i])) + val_NAG
    # val = val + pd.to_numeric(d_OAG[d_OAG.get(d_str[i])])
    else:
        continue
print('cumulate value NAG:', val_NAG)

'''ML Coding on CAG category starts'''
for i in range(0, len(d_str)):
    if d_CAG.get(d_str[i], 'NA') != 'NA':
        val_CAG = pd.to_numeric(d_CAG.get(d_str[i])) + val_CAG
    # val = val + pd.to_numeric(d_OAG[d_OAG.get(d_str[i])])
    else:
        continue
print('cumulate value CAG:', val_CAG)

'''ML REsults'''

if val_NAG >= val_OAG:
    if val_NAG >= val_CAG:
        print('Given Text belongs to NAG category')
    else:
        print('Given Text belongs to CAG category')
elif val_OAG >= val_CAG:
    print('Given Text belongs to OAG category')
else:
    print('Given Text belongs to CAG category')

print('done')

print(d_str)
