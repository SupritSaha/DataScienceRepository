# Author: Suprit Saha <supritster@gmail.com>
# Date : 19-02-2019
# Work in progress

import numpy as np
import pandas as pd
import time
from contextlib import contextmanager
from sklearn.preprocessing import LabelEncoder

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')

def load_csv_data(train_path,test_path,header=True):
    """
    Reads csv train and test files

    Parameters
    ---------------------------------------------------
    train_path : Train filepath
    test_path : Test filepath

    Returns
    ---------------------------------------------------
    train : Train pandas data frame
    test : Test pandas data frame
    """
    with timer("Reading input files"):
        try:
            train = pd.read_csv(train_path)
            test = pd.read_csv(test_path)
        except:
            print("File not in csv format !")
    return train, test


def remove_constant_columns(train_raw,test_raw):
    """
    Removes constant columns from dataset

    Parameters
    ---------------------------------------------------
    train_raw : Train pandas data frame
    test_raw : Test pandas data frame

    Returns
    ---------------------------------------------------
    train : Train pandas data frame without constant columns
    test : Test pandas data frame without constant columns
    """
    with timer("Removing constant columns"):
        train = pd.DataFrame.copy(train_raw)
        test = pd.DataFrame.copy(test_raw)
        n_train = train.shape[0]
        full = pd.concat([train,test],axis=0,ignore_index=True)
        n_full = full.shape[0]

        constant_columns = []
        for colname in full.columns:
            if len(np.unique(full[colname].values.astype('str'))) == 1:
                constant_columns.append(colname)
                del full[colname]
        print('Constant columns :', constant_columns)
        train = full.iloc[0:n_train,:].reset_index(drop=True)
        test = full.iloc[n_train:n_full,:].reset_index(drop=True)

    return train, test

def cyclical_transform(x):
    return np.sin(2 * np.pi * x/np.max(x)),np.cos(2 * np.pi * x/np.max(x))

def create_date_features(date_cols,train_raw,test_raw,keep=True,cyclical=False):
    """
    Creates date features

    Parameters
    ---------------------------------------------------
    date_cols : List of date columns
    train_raw : Train pandas data frame
    test_raw : Test pandas data frame
    cyclical : boolean

    Returns
    ---------------------------------------------------
    train : Train pandas data frame with date features(cyclical features if True)
    test : Test pandas data frame with date features(cyclical features if True)

    References
    ---------------------------------------------------
    https://www.kaggle.com/avanwyk/encoding-cyclical-features-for-deep-learning
    """
    with timer("Creating date features"):
        train = pd.DataFrame.copy(train_raw)
        test = pd.DataFrame.copy(test_raw)
        n_train = train.shape[0]
        full = pd.concat([train,test],axis=0,ignore_index=True)
        n_full = full.shape[0]

        for colname in date_cols:
            full[colname] = pd.DatetimeIndex(full[colname])
            full[colname+"_year"] = full[colname].dt.year
            full[colname+"_month"] = full[colname].dt.month
            full[colname+"_day"] = full[colname].dt.day
            full[colname+"_weekday"] = full[colname].dt.weekday
            if keep==False:
                full.drop(colname, axis=1, inplace=True)
            if cyclical==True:
               full[colname+"_month"+"_sin"], full[colname+"_month"+"_cos"] = cyclical_transform(full[colname+"_month"])
               full[colname+"_day"+"_sin"], full[colname+"_day"+"_cos"] = cyclical_transform(full[colname+"_day"])
               full[colname+"_weekday"+"_sin"] , full[colname+"_weekday"+"_cos"] = cyclical_transform(full[colname+"_weekday"])
               del full[colname+"_month"],full[colname+"_day"],full[colname+"_weekday"]

        train = full.iloc[0:n_train,:].reset_index(drop=True)
        test = full.iloc[n_train:n_full,:].reset_index(drop=True)

    return train, test

def one_hot_encoding(train_raw,test_raw,columns):
    """
    Creates one hot encoded

    Parameters
    ---------------------------------------------------
    train_raw : Train pandas data frame
    test_raw : Test pandas data frame
    columns : List of columns to be one hot encoded

    Returns
    ---------------------------------------------------
    train : Train pandas data frame with ohe features
    test : Test pandas data frame with ohe features
    """
    with timer("Creating one hot encoded features"):
        train = pd.DataFrame.copy(train_raw)
        test = pd.DataFrame.copy(test_raw)
        n_train = train.shape[0]
        full = pd.concat([train,test],axis=0,ignore_index=True,sort=True)
        n_full = full.shape[0]
    full = pd.get_dummies(full,drop_first=True))
    train = full.iloc[0:n_train,:].reset_index(drop=True)
    test = full.iloc[n_train:n_full,:].reset_index(drop=True)

    return train, test

def label_encoding(train_raw,test_raw,columns):
    """
    Creates label encoded features

    Parameters
    ---------------------------------------------------
    train_raw : Train pandas data frame
    test_raw : Test pandas data frame
    columns : List of columns to be LabelEncoded

    Returns
    ---------------------------------------------------
    train : Train pandas data frame with LabelEncoded features
    test : Test pandas data frame with LabelEncoded features
    """
    with timer("Creating label encoded features"):
        train = pd.DataFrame.copy(train_raw)
        test = pd.DataFrame.copy(test_raw)
        n_train = train.shape[0]
        full = pd.concat([train,test],axis=0,ignore_index=True,sort=True)
        n_full = full.shape[0]

    for colname in columns:
        full[colname] = LabelEncoder().fit_transform(full[colname].astype('str'))

    train = full.iloc[0:n_train,:].reset_index(drop=True)
    test = full.iloc[n_train:n_full,:].reset_index(drop=True)

    return train, test


def frequency_encoding(train_raw,test_raw,columns):
    """
    Creates frequency encoded features

    Parameters
    ---------------------------------------------------
    train_raw : Train pandas data frame
    test_raw : Test pandas data frame
    columns : List of columns to be LabelEncoded

    Returns
    ---------------------------------------------------
    train : Train pandas data frame with frequency features
    test : Test pandas data frame with frequency features
    """
    with timer("Creating frequency encoded features"):
        train = pd.DataFrame.copy(train_raw)
        test = pd.DataFrame.copy(test_raw)
        n_train = train.shape[0]

    for colname in columns:
        freq_count = train.groupby(colname)[colname].count()/n_train
        train[colname+'_freq'] = train[colname].map(freq_count)
        test[colname+'_freq'] = test[colname].map(freq_count)
        del train[colname], test[colname]

    return train, test


def mean_encoding(train_raw,test_raw,columns,target_column,alpha=0,add_random=False, rmean=0, rstd=0.1):
    """
    Creates mean encoded features

    Parameters
    ---------------------------------------------------
    train_raw : Train pandas data frame
    test_raw : Test pandas data frame
    columns : List of columns to be encoded
    target_column : Str type target variable name
    alpha : numeric regularisation parameter
    add_random : boolean to add random noise to raw encodings
    rmean : random normal noise mean
    rstd : random normal noise std

    Returns
    ---------------------------------------------------
    train : Train pandas data frame with mean features
    test : Test pandas data frame with mean features
    """
    with timer("Creating target mean encoded features"):
        train = pd.DataFrame.copy(train_raw)
        test = pd.DataFrame.copy(test_raw)
        n_train = train.shape[0]
        target_overall_mean = train[target_column].mean()

    for colname in columns:
        count_category = train.groupby(colname)[target_column].count()
        avg_category = train.groupby(colname)[target_column].mean()
        smoothed_avg_category = (avg_category*count_category + target_overall_mean*alpha)/(count_category + alpha)
        train[colname+'_smoothed_avg'] = train[colname].map(smoothed_avg_category)
        if add_random:
            train[colname+'_smoothed_avg'] = train[colname+'_smoothed_avg'] + np.random.normal(loc=rmean, scale=rstd, size=n_train)
        test[colname+'_smoothed_avg'] = test[colname].map(smoothed_avg_category)
        del train[colname], test[colname]

    return train, test

def capping_flooring(column,upper=0.99,lower=0.01):
    """
    Capping and flooring of variable

    Parameters
    ---------------------------------------------------
    column : list, tuple, array-like or pandas Series
    upper : upper percentile value
    lower : lower percentile value

    Returns
    ---------------------------------------------------
    column : capped and floored pandas series
    """
    if isinstance(column,(list,tuple)):
        column = pd.Series(column)
    percentiles = column.quantile([lower,upper]).values
    column[column < percentiles[0]] = percentiles[0]
    column[column > percentiles[1]] = percentiles[1]
    return column
