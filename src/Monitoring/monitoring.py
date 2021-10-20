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
    drift_tests = ["kolmogorov_smirnov", "t_test", "levene_test"]
    for test in drift_tests:
        function_test= globals()[test]
        apply_test(df, batch, function_test)