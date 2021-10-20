# -*- coding: utf-8 -*-
from scipy import stats
import logging
import utils as u
logger = logging.getLogger('main_logger')

from scipy import stats
import numpy as np

def create_batches(df,n):
    return np.array_split(df,n)


def kolmogorov_smirnov(data_1, data_2):
    # test whether two columns have the same distribution
    _, pvalue = stats.ks_2samp(data_1, data_2)
    return pvalue >= 0.05

def t_test(data_1, data_2):
    # test whether two columns have the same mean
    _, pvalue = stats.ttest_ind(data_1, data_2, axis=0, equal_var=False)
    return pvalue >= 0.05

def levene_test(data_1, data_2):
    # test whether two columns have the same variance
    _, pvalue = stats.levene(data_1, data_2)
    return pvalue >= 0.05

def check_set_columns (data_1, data_2):
    if set(data_1.columns) == set(data_2.columns):
        err_msg = "\t ->Format check_set_columns: OK,"
    else :
        err_msg = "\t ->Format check_set_columns: NOT OK, the new batch do not have the same columns"
    print(err_msg)
    return not(set(data_1.columns) == set(data_2.columns))

def check_nb_nan (data_1, data_2):
    percent_missing_1 = data_1.isnull().sum() * 100 / len(data_1)
    percent_missing_2 = data_2.isnull().sum() * 100 / len(data_2)
    if (percent_missing_1+percent_missing_2==0):
        return False
    diff_in_percent = abs(percent_missing_1 - percent_missing_2) / (percent_missing_1+percent_missing_2)
    return diff_in_percent> 0.01


def psi(data_1, data_2, buckettype='bins', buckets=10):
    def scale_range(input, min, max):
        input += -(np.min(input))
        input /= np.max(input) / (max - min)
        input += min
        return input

    breakpoints = np.arange(0, buckets + 1) / (buckets) * 100

    if buckettype == 'bins':
        breakpoints = scale_range(breakpoints, np.min(data_1), np.max(data_1))
    elif buckettype == 'quantiles':
        breakpoints = np.stack([np.percentile(data_1, b) for b in breakpoints])

    expected_percents = np.histogram(data_1, breakpoints)[0] / len(data_1)
    actual_percents = np.histogram(data_2, breakpoints)[0] / len(data_2)

    def sub_psi(e_perc, a_perc):
        '''Calculate the actual PSI value from comparing the values.
            Update the actual value to a very small number if equal to zero
        '''
        if a_perc == 0:
            a_perc = 0.0001
        if e_perc == 0:
            e_perc = 0.0001

        value = (e_perc - a_perc) * np.log(e_perc / a_perc)
        return (value)

    psi_value = np.sum(sub_psi(expected_percents[i], actual_percents[i]) for i in range(0, len(expected_percents)))

    return psi_value>0.5


def apply_test(df, batch, test):
    int_col = u.get_numerical_columns(df)
    results = {}
    print("\t ->" + test.__name__ +":")
    for col in int_col:
        data_1 = df.loc[:, col]
        data_2 = batch.loc[:, col]
        results[col] = test(data_1, data_2)
        if (results[col]):
            print("\t \t -"+col)

    return results

def main_monitoring(df,batches):

    count=1
    for batch in batches:
        print("batch %d" %count)
        main_monitoring_batch(df,batch)
        count+=1
        print("**************************************** \n")

def main_monitoring_batch(df,batch):
    check_set_columns(df,batch)
    tests = ["check_nb_nan","kolmogorov_smirnov", "t_test", "levene_test","psi"]
    for test in tests:
        function_test= globals()[test]
        apply_test(df, batch, function_test)

