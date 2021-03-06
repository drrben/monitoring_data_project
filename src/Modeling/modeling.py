# -*- coding: utf-8 -*-

import logging

from scipy.stats import loguniform

logger = logging.getLogger('main_logger')
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import f1_score, fbeta_score, make_scorer
from sklearn.model_selection import GridSearchCV, KFold

# What can be done in this script: (not mandatory)
# Implementing a cross validation (function) to check the robustness of a parameter set
# Implementing on it a grid search to select the best parameter set
# Implementing other models



def main_modeling_from_name(X_train,y_train, conf):
    """
    Main modeling function: it launches a grid search using the correct model according to the conf file
    Args:
        X_train: X_train
        y_train: y_train
        conf: configuration file

    Returns: model fitted on the train set and its best params

    """

    dict_function_GS_params = {
        'random_forest': 'get_GS_params_RFClassifier',
        'lightgbm': 'get_GS_params_lightgbm',
        'logistic_regression': 'get_GS_params_LRClassifier',
        'ridge_classifier': 'get_GS_params_RidgeClassifier',
    }
    dict_function_train_model = {
        'random_forest': 'train_RFClassifier',
        'lightgbm': 'train_lightgbm',
        'logistic_regression': 'train_LRClassifier',
        'ridge_classifier': 'train_RidgeClassifier',
    }

    selected_model = conf['selected_model']
    function_get_GS_params = globals()[dict_function_GS_params[selected_model]]
    estimator, params_grid = function_get_GS_params()

    logger.info('Beginning of Grid Search using ' + selected_model)
    best_params, best_score = main_GS_from_estimator_and_params(X_train, y_train, estimator, params_grid, conf)

    function_train = globals()[dict_function_train_model[selected_model]]
    model = function_train(X_train,y_train, best_params)
    logger.info('Enfd of Grid Search using ' + selected_model)
    logger.info('Best parameteres are :')
    logger.info(best_params)
    logger.info('best score' + str(best_score))

    return model, best_params


def main_GS_from_estimator_and_params(X_train,y_train, estimator, params_grid, conf):
    """
    Main function to run a grid search
    Args:
        X_train: X_train
        y_train:  y_train
        estimator: unfit model to use
        params_grid: grid search to run
        conf: conf file

    Returns: best params and score achieved in the GS

    """
    gkf = KFold(n_splits=3, shuffle=True, random_state=42).split(X=X_train, y=y_train)


    gsearch = GridSearchCV(estimator=estimator, param_grid=params_grid, cv=gkf,
                           scoring=make_scorer(f1_score), verbose=1, n_jobs = -1)
    best_model = gsearch.fit(X=X_train, y=y_train)

    means = gsearch.cv_results_['mean_test_score']
    stds = gsearch.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, gsearch.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

    return best_model.best_params_, best_model.best_score_


####### LIGHTGBM  ########

def get_GS_params_lightgbm():
    """
    Gives params and models to use for the grid_search using LightGBM Classifier
    Returns:Estimator and params for the grid_search

    """
    params_grid = {'objective': ['binary'],
    'max_depth': [6,100],
    'reg_alpha': [0, 0.1],
    'min_data_in_leaf': [5, 10],
    'learning_rate': [0.01],
    'scale_pos_weight': [0.2, 1, 3, 10]
    }

    estimator = lgb.LGBMClassifier(boosting_type='gbdt',  objective='binary', num_boost_round=200, learning_rate=0.01)

    return estimator, params_grid

def train_lightgbm(X_train,y_train,params):
    """
    Training function for a Lightgbm
    Args:
        X_train: X_train
        y_train: y_train
        params: params to use for the fitting

    Returns: trained lightgbm model

    """
    dftrainLGB = lgb.Dataset(data=X_train, label=y_train, feature_name=list(X_train))
    model = lgb.train(params,dftrainLGB)
    return model


####### RANDOM FOREST  ########
def get_GS_params_RFClassifier():
    """
    Gives params and models to use for the grid_search using Random Forest Classifier
    Returns:Estimator and params for the grid_search
    """
    params_grid = {'bootstrap': [True],
              'criterion': ['entropy'],
              'max_depth': [3,6],
              'max_features': [3,10],
              'min_samples_leaf': [4],
              'min_samples_split': [3]}
    estimator = RandomForestClassifier()

    return estimator, params_grid

def train_RFClassifier(X_train,y_train,params):
    """
    Training function for a random forest Classifier
    Args:
        X_train: 
        y_train: 
        params: params to use for the fitting

    Returns: trained random forest model

    """
    model = RandomForestClassifier(**params).fit(X_train,y_train)
    return model

####### LOGISTIC REGRESSION  ########
def get_GS_params_LRClassifier():
    """
    Gives params and models to use for the grid_search using a Logistic Regression
    Returns:Estimator and params for the grid_search
    """
    params_grid = {
        "solver": ["liblinear"],
        'penalty': ['l1', 'l2'],
        'C': [0.001, .009, 0.01, .09, 1, 5, 10, 25]
    }

    estimator = LogisticRegression()

    return estimator, params_grid

def train_LRClassifier(X_train,y_train,params):
    """
    Training function for a a Logistic Regression
    Args:
        X_train: 
        y_train: 
        params: params to use for the fitting

    Returns: trained random forest model

    """
    model = LogisticRegression(**params).fit(X_train,y_train)
    return model

####### RIDGE CLASSIFICATION  ########
def get_GS_params_RidgeClassifier():
    """
    Gives params and models to use for the grid_search using a Ridge Classifier
    Returns:Estimator and params for the grid_search
    """
    params_grid = {
        "alpha": [1, 10]
    }

    estimator = RidgeClassifier()

    return estimator, params_grid

def train_RidgeClassifier(X_train,y_train,params):
    """
    Training function for a a Ridge Classifier
    Args:
        X_train: 
        y_train: 
        params: params to use for the fitting

    Returns: trained random forest model

    """
    model = RidgeClassifier(**params).fit(X_train,y_train)
    return model

