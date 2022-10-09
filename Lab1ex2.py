# If required, download the dataset

from __future__ import division
import requests
import os.path
import zipfile
if (not os.path.isdir('./HAPT Data Set')):
    open('./HAPT Data Set.zip', 'wb').write(requests.get(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00341/HAPT%20Data%20Set.zip", 
        allow_redirects=True).content)
    zipfile.ZipFile('./HAPT Data Set.zip', 'r').extractall('./HAPT Data Set')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

import random
random.seed(7)

# display pandas results to 3 decimal points, not in scientific notation
# pd.set_option('display.float_format', lambda x: '%.3f' % x)

# get number of features
with open('./HAPT Data Set/features.txt') as f:
    features = f.read().split()

print('There are {} features.'.format(len(features)))

# get number of activities
with open('./HAPT Data Set/activity_labels.txt') as f:
    activity_labels = f.readlines()

activity_df = [x.split() for x in activity_labels]
print('There are {} activities.'.format(len(activity_df)))
pd.DataFrame(activity_df, columns = ['Activity_id', 'Activity_label'])

# Defining training and testing sets:

# The iloc() function in python is defined in the Pandas module that helps us to select a specific row or column from the data set.
X_train = pd.read_table('./HAPT Data Set/Train/X_train.txt',
             header = None, sep = " ", names = list(dict.fromkeys(features)))
X_train.iloc[:10, :10].head()

y_train = pd.read_table('./HAPT Data Set/Train/y_train.txt',
             header = None, sep = " ", names = ['Activity_id'])
y_train.head()

X_test = pd.read_table('./HAPT Data Set/Test/X_test.txt',
             header = None, sep = " ", names = list(dict.fromkeys(features)))
y_test = pd.read_table('./HAPT Data Set/Test/y_test.txt',
             header = None, sep = " ", names = ['Activity_id'])

# Apply linear SVC
from sklearn.svm import LinearSVC
from sklearn.model_selection import validation_curve

# Declare the hyper-parameter
C_params = np.logspace(-6, 3, 10)

# Declare the classfier
clf_svc = LinearSVC(random_state = 7)

# Compute training and test scores for varying parameter values given a classifier object
# cv: Determines the cross-validation splitting strategy. 
# train_scores: scores of training set
# val_scores: scores of testing set
train_scores, val_scores = validation_curve(
    clf_svc, X_train.values, y_train.values.flatten(),
    param_name = "C", param_range = C_params,
    cv = 5, scoring = "accuracy", n_jobs = -1)