import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, roc_auc_score, classification_report, log_loss, make_scorer
from scipy.stats import ttest_ind


def conversion(df):
    df = df.copy()
    df.last_trip_date = pd.to_datetime(df.last_trip_date, format='%Y-%m-%d')
    df.signup_date = pd.to_datetime(df.signup_date, format='%Y-%m-%d')
    return df


def fix_missing(df):
    df = df.copy()
    df.phone = df.phone.fillna('No Phone')
    df = df[~df.avg_rating_by_driver.isna()]
    return df


def feature_engineering(df):
    df = df.copy()
    df['bin_avg_rating_by_driver'] = pd.cut(df.avg_rating_by_driver, bins=[0., 2.99, 3.99, 4.99, 5],
                                            include_lowest=True, right=True)
    df['bin_avg_rating_of_driver'] = pd.cut(df.avg_rating_of_driver, bins=[0., 2.99, 3.99, 4.99, 5],
                                            include_lowest=True, right=True)
    df.bin_avg_rating_by_driver.cat.add_categories('missing', inplace=True)
    df.bin_avg_rating_of_driver.cat.add_categories('missing', inplace=True)
    df.bin_avg_rating_by_driver.fillna('missing', inplace=True)
    df.bin_avg_rating_of_driver.fillna('missing', inplace=True)
    cutoff_date = pd.to_datetime('2014-07-01', format='%Y-%m-%d') - datetime.timedelta(30, 0, 0)
    df['churned'] = df.last_trip_date < cutoff_date
    return df


def create_indicator_columns(df):
    df = df.copy()
    phones = pd.get_dummies(df.phone)
    df[phones.columns.to_list()] = phones
    cities = pd.get_dummies(df.city)
    df[cities.columns.to_list()] = cities
    by_driver = pd.get_dummies(df.bin_avg_rating_by_driver, prefix='by_drv')
    df[by_driver.columns.to_list()] = by_driver
    of_driver = pd.get_dummies(df.bin_avg_rating_of_driver, prefix='of_drv')
    df[of_driver.columns.to_list()] = of_driver
    return df


def eda(df):
    df = conversion(df)
    df = fix_missing(df)
    df = feature_engineering(df)
    df = create_indicator_columns(df)
    return df


def split(df):
    y = df.churned
    X = df.drop(columns=['churned', 'avg_rating_by_driver', 'avg_rating_of_driver', 'city', 'phone', 'last_trip_date', 'signup_date', 'bin_avg_rating_by_driver', 'bin_avg_rating_of_driver'])
    return X, y


def ttest_by(vals, by):
    vals1 = vals[by]
    vals2 = vals[-by]
    return ttest_ind(vals1, vals2)
