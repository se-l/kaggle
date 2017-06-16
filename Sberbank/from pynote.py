import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import getpass
color = sns.color_palette()
import os
import datetime
if os.name == 'posix':
    projectDir = r'/home/' + getpass.getuser() + r'/repos/kaggle/Sberbank'
else:
    projectDir = r'C:/repos/kaggle/Sberbank'

from sklearn import model_selection, preprocessing
import xgboost as xgb
import lightgbm as lgb
import logging as Logger
from utils.utils import dotdict, Logger
import argparse
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

def parse():
    parser = argparse.ArgumentParser(description='Pattern finder - GBM')
    parser.add_argument('-srcargs', type=int, action='store', help='Curtail dataset for quick run?', default=1)
    parser.add_argument('-reduceData', type=int, action='store', help='Curtail dataset for quick run?', default=1)
    parser.add_argument('-paramset', type=int, action='store', help='Curtail dataset for quick run?', default=2)
    parser.add_argument('-algo', action='store', help='Instrument subscription file.', default='xgb')
    parser.add_argument('-xgbcv', type=int, action='store', help='Instrument subscription file.', default=0)

    args = parser.parse_args()
    if args.srcargs == 1:
        argss = ['-reduceData', '0',
                 '-paramset', '2',
                 '-algo', 'xgb',
                 '-xgbcv', '0']
        args = parser.parse_args(argss)
    return args

def initLogger():
    Logger.init_log(os.path.join(projectDir,'log/xgblog-{}'.format(datetime.date.today())))

def loadData():
    # train_df = pd.read_csv(os.path.join(projectDir,"input/train.csv"))
    # train_df.shape
    train_df = pd.read_csv(os.path.join(projectDir,"input/train.csv"), parse_dates=['timestamp'])
    test_df = pd.read_csv(os.path.join(projectDir, "input/test.csv"), parse_dates=['timestamp'])
    return train_df, test_df

def prepareData1(traind_df):

    train_df['price_doc_log'] = np.log1p(train_df['price_doc'])

    train_na = (train_df.isnull().sum() / len(train_df)) * 100
    train_na = train_na.drop(train_na[train_na == 0].index).sort_values(ascending=False)

    train_df['state'].value_counts()
    train_df.loc[train_df['state']==33, 'state'] = train_df['state'].mode().iloc[0]
    # train_df['state'].value_counts()
    # train_df['build_year'].value_counts()
    train_df.loc[train_df['build_year'] == 20052009, 'build_year'] = 2007
    #build_year has an erronus value 20052009. Since its unclear which it should be, let's replace with 2007
    train_1900= train_df[train_df['build_year'] < 1900]
    # train_1900.shape
    internal_chars = ['full_sq', 'life_sq', 'floor', 'max_floor', 'build_year', 'num_room', 'kitch_sq', 'state', 'price_doc']
    corrmat = train_df[internal_chars].corr()


    import matplotlib.dates as mdates

    years = mdates.YearLocator()   # every year
    yearsFmt = mdates.DateFormatter('%Y')
    ts_vc = train_df['timestamp'].value_counts()

    ## Demographic Characteristics
    demo_vars = ['area_m', 'raion_popul', 'full_all', 'male_f', 'female_f', 'young_all', 'young_female',
                 'work_all', 'work_male', 'work_female', 'price_doc']

    train_df['area_km'] = train_df['area_m'] / 1000000
    train_df['density'] = train_df['raion_popul'] / train_df['area_km']
    train_df['work_share'] = train_df['work_all'] / train_df['raion_popul']
    school_chars = ['children_preschool', 'preschool_quota', 'preschool_education_centers_raion', 'children_school',
                    'school_quota', 'school_education_centers_raion', 'school_education_centers_top_20_raion',
                    'university_top_20_raion', 'additional_education_raion', 'additional_education_km', 'university_km', 'price_doc']

    train_df['university_top_20_raion'].unique()
    #Cultural/Recreational Characteristics
    # cult_chars = ['sport_objects_raion', 'culture_objects_top_25_raion', 'shopping_centers_raion', 'park_km', 'fitness_km',
    #                 'swim_pool_km', 'ice_rink_km','stadium_km', 'basketball_km', 'shopping_centers_km', 'big_church_km',
    #                 'church_synagogue_km', 'mosque_km', 'theater_km', 'museum_km', 'exhibition_km', 'catering_km', 'price_doc']
    # corrmat = train_df[cult_chars].corr()
    # train_df.groupby('culture_objects_top_25')['price_doc'].median()

    #Variable Importance
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import LabelEncoder
    X_train = train_df.drop(labels=['timestamp', 'id', 'incineration_raion'], axis=1).dropna()
    y_train = X_train['price_doc']
    X_train.drop('price_doc', axis=1, inplace=True)
    for f in X_train.columns:
        if X_train[f].dtype == 'object':
            lbl = LabelEncoder()
            lbl.fit(X_train[f])
            X_train[f] = lbl.transform(X_train[f])
    rf = RandomForestRegressor(random_state=0)
    rf = rf.fit(X_train, y_train)

    #Train vs Test Data
    test_na = (test_df.isnull().sum() / len(test_df)) * 100
    test_na = test_na.drop(test_na[test_na == 0].index).sort_values(ascending=False)

    all_data = pd.concat([train_df.drop('price_doc', axis=1), test_df])
    all_data['dataset'] = ''
    l = len(train_df)
    all_data.iloc[:l]['dataset'] = 'train'
    all_data.iloc[l:]['dataset'] = 'test'
    train_dataset = all_data['dataset'] == 'train'

    years = mdates.YearLocator()   # every year
    yearsFmt = mdates.DateFormatter('%Y')

train_df, test_df = loadData()
macro_df = pd.read_csv(os.path.join(projectDir,"input/macro.csv"), parse_dates=['timestamp'])


def prepareData2(train_df, test_df, macro_df):
    train_df = pd.merge(train_df, macro_df, how='left', on='timestamp')
    test_df = pd.merge(test_df, macro_df, how='left', on='timestamp')
    print(train_df.shape, test_df.shape)

    # truncate the extreme values in price_doc #
    ulimit = np.percentile(train_df.price_doc.values, 99)
    llimit = np.percentile(train_df.price_doc.values, 1)
    train_df['price_doc'].ix[train_df['price_doc']>ulimit] = ulimit
    train_df['price_doc'].ix[train_df['price_doc']<llimit] = llimit

    for f in train_df.columns:
        if train_df[f].dtype=='object':
            # print(f)
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train_df[f].values.astype('str')) + list(test_df[f].values.astype('str')))
            train_df[f] = lbl.transform(list(train_df[f].values.astype('str')))
            test_df[f] = lbl.transform(list(test_df[f].values.astype('str')))

    # year and month #
    train_df["yearmonth"] = train_df["timestamp"].dt.year*100 + train_df["timestamp"].dt.month
    test_df["yearmonth"] = test_df["timestamp"].dt.year*100 + test_df["timestamp"].dt.month

    # year and week #
    train_df["yearweek"] = train_df["timestamp"].dt.year*100 + train_df["timestamp"].dt.weekofyear
    test_df["yearweek"] = test_df["timestamp"].dt.year*100 + test_df["timestamp"].dt.weekofyear

    # year #
    train_df["year"] = train_df["timestamp"].dt.year
    test_df["year"] = test_df["timestamp"].dt.year

    # month of year #
    train_df["month_of_year"] = train_df["timestamp"].dt.month
    test_df["month_of_year"] = test_df["timestamp"].dt.month

    # week of year #
    train_df["week_of_year"] = train_df["timestamp"].dt.weekofyear
    test_df["week_of_year"] = test_df["timestamp"].dt.weekofyear

    # day of week #
    train_df["day_of_week"] = train_df["timestamp"].dt.weekday
    test_df["day_of_week"] = test_df["timestamp"].dt.weekday

    # ratio of living area to full area #
    train_df["ratio_life_sq_full_sq"] = train_df["life_sq"] / np.maximum(train_df["full_sq"].astype("float"),1)
    test_df["ratio_life_sq_full_sq"] = test_df["life_sq"] / np.maximum(test_df["full_sq"].astype("float"),1)
    train_df["ratio_life_sq_full_sq"].ix[train_df["ratio_life_sq_full_sq"]<0] = 0
    train_df["ratio_life_sq_full_sq"].ix[train_df["ratio_life_sq_full_sq"]>1] = 1
    test_df["ratio_life_sq_full_sq"].ix[test_df["ratio_life_sq_full_sq"]<0] = 0
    test_df["ratio_life_sq_full_sq"].ix[test_df["ratio_life_sq_full_sq"]>1] = 1

    # ratio of kitchen area to living area #
    train_df["ratio_kitch_sq_life_sq"] = train_df["kitch_sq"] / np.maximum(train_df["life_sq"].astype("float"),1)
    test_df["ratio_kitch_sq_life_sq"] = test_df["kitch_sq"] / np.maximum(test_df["life_sq"].astype("float"),1)
    train_df["ratio_kitch_sq_life_sq"].ix[train_df["ratio_kitch_sq_life_sq"]<0] = 0
    train_df["ratio_kitch_sq_life_sq"].ix[train_df["ratio_kitch_sq_life_sq"]>1] = 1
    test_df["ratio_kitch_sq_life_sq"].ix[test_df["ratio_kitch_sq_life_sq"]<0] = 0
    test_df["ratio_kitch_sq_life_sq"].ix[test_df["ratio_kitch_sq_life_sq"]>1] = 1

    # ratio of kitchen area to full area #
    train_df["ratio_kitch_sq_full_sq"] = train_df["kitch_sq"] / np.maximum(train_df["full_sq"].astype("float"),1)
    test_df["ratio_kitch_sq_full_sq"] = test_df["kitch_sq"] / np.maximum(test_df["full_sq"].astype("float"),1)
    train_df["ratio_kitch_sq_full_sq"].ix[train_df["ratio_kitch_sq_full_sq"]<0] = 0
    train_df["ratio_kitch_sq_full_sq"].ix[train_df["ratio_kitch_sq_full_sq"]>1] = 1
    test_df["ratio_kitch_sq_full_sq"].ix[test_df["ratio_kitch_sq_full_sq"]<0] = 0
    test_df["ratio_kitch_sq_full_sq"].ix[test_df["ratio_kitch_sq_full_sq"]>1] = 1

    # floor of the house to the total number of floors in the house #
    train_df["ratio_floor_max_floor"] = train_df["floor"] / train_df["max_floor"].astype("float")
    test_df["ratio_floor_max_floor"] = test_df["floor"] / test_df["max_floor"].astype("float")

    # num of floor from top #
    train_df["floor_from_top"] = train_df["max_floor"] - train_df["floor"]
    test_df["floor_from_top"] = test_df["max_floor"] - test_df["floor"]

    train_df["extra_sq"] = train_df["full_sq"] - train_df["life_sq"]
    test_df["extra_sq"] = test_df["full_sq"] - test_df["life_sq"]

    train_df["age_of_building"] = train_df["build_year"] - train_df["year"]
    test_df["age_of_building"] = test_df["build_year"] - test_df["year"]

    train_df = add_count(train_df, "yearmonth")
    test_df = add_count(test_df, "yearmonth")

    train_df = add_count(train_df, "yearweek")
    test_df = add_count(test_df, "yearweek")

    train_df["ratio_preschool"] = train_df["children_preschool"] / train_df["preschool_quota"].astype("float")
    test_df["ratio_preschool"] = test_df["children_preschool"] / test_df["preschool_quota"].astype("float")

    train_df["ratio_school"] = train_df["children_school"] / train_df["school_quota"].astype("float")
    test_df["ratio_school"] = test_df["children_school"] / test_df["school_quota"].astype("float")

    y_train = train_df["price_doc"]
    x_train = train_df.drop(["id", "timestamp", "price_doc"], axis=1)
    x_test = test_df.drop(["id", "timestamp"], axis=1)

    test_id=test_df.id

    return x_train, y_train, x_test, test_id

def add_count(df, group_col):
    grouped_df = df.groupby(group_col)["id"].aggregate("count").reset_index()
    grouped_df.columns = [group_col, "count_"+group_col]
    df = pd.merge(df, grouped_df, on=group_col, how="left")
    return df

def runAlgo(x_train, y_train, x_test, test_id):

    if args.algo == 'xgb':

        xgb_params = {
            'eta': 0.05,
            'max_depth': 5,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'objective': 'reg:linear',
            'eval_metric': 'rmse',
            'silent': 1
        }

        dtrain = xgb.DMatrix(x_train, y_train)
        dtest = xgb.DMatrix(x_test)

        cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000,
                           early_stopping_rounds=20,
            verbose_eval=50, show_stdv=False)
        cv_output[['train-rmse-mean', 'test-rmse-mean']].plot()

        num_boost_rounds = len(cv_output)
        model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round= num_boost_rounds)
        ts = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        model.save_model(os.path.join(projectDir, 'model/{}.model'.format(ts)))

        fig, ax = plt.subplots(1, 1, figsize=(8, 13))
        xgb.plot_importance(model, max_num_features=50, height=0.5, ax=ax)

        y_predict = model.predict(dtest)

        output = pd.DataFrame({'id': test_id, 'price_doc': y_predict})
        output.to_csv(os.path.join(projectDir,'subm/xgbSub.csv'), index=False)

    if args.algo == 'lgb':
        train_data = lgb.Dataset(dTrain, label=yTrain)
        test_data = lgb.Dataset(dTest, label=yTest, reference=train_data)

        if True:
            # setting parameters for lightgbm
            param = {
                'learning_rate': [0.01],
                # 'n_estimators': [100],
                # 'num_round': [200],
                'num_leaves': [255],
                # 'min_data_in_leaf': [500],
                'max_bin': [2500],
                'max_depth': [-1],
                'boosting_type': ['gbdt'],
                'objective': ['binary'],
                'seed': [777],
                # 'colsample_bytree': [0.8],
                # 'subsample': [0.8],
                # 'reg_alpha': [0, 0.1, 0.5, 1],
                # 'reg_lambda': [0, 1, 2, 5, 6, 7, 9, 10],
                # 'is_unbalance': [True, False],
                'nthread': [-1],
                'metric': ['auc', 'binary_logloss'],
            }
            # Here we have set max_depth in xgb and LightGBM to 7 to have a fair comparison between the two.
            # training our model using light gbm
            # num_round = 500
            start = datetime.datetime.now()
            lgbm = lgb.train(param, train_data,
                             valid_sets=[train_data, test_data],
                             # 'eval_metric': 'binary_log_loss',
                             early_stopping_rounds=5)
            # cvresult = lgb.cv(param, train_data, num_round, nfold=5)
            stop = datetime.datetime.now()
            # Execution time of the model
            execution_time_lgbm = stop - start
            execution_time_lgbm
            # predicting on test set
            ypred = lgbm.predict(dTest, num_iteration=lgbm.best_iteration)

            # converting probabilities into 0 or 1
            predBnry = [1 if x > 0.5 else 0 for x in ypred]
            accuracy = accuracy_score(yTest, predBnry)
            Logger.info("Accuracy : %.2f%%" % (accuracy * 100.0))

        if False:
            estimator = lgb.LGBMClassifier(nthread=-1,
                                           silent=False)  # ,categorical_feature=[list(X_train).index(catFeature) for catFeature in categorical_features])
            param_grid = {
                'learning_rate': [0.01],
                # 'n_estimators': [100],
                # 'num_round': [200],
                'num_leaves': [255],
                'min_data_in_leaf': [500],
                'max_bin': [5000],
                'max_depth': [-1],
                'boosting_type': ['gbdt'],
                'objective': ['binary'],
                'seed': [777],
                # 'colsample_bytree': [0.8],
                # 'subsample': [0.8],
                # 'reg_alpha': [0, 0.1, 0.5, 1],
                # 'reg_lambda': [0, 1, 2, 5, 6, 7, 9, 10],
                # 'is_unbalance': [True, False],
                'nthread': [-1],
            }
            fit_params = None
            fit_params = {
                # 'num_round': param_grid['num_round'],
                'eval_set': [(dTrain, yTrain), (dTest, yTest)],
                # 'eval_metric': 'binary_log_loss',
                'early_stopping_rounds': 5
            }
            ngrid = 1
            gbm = RandomizedSearchCV(estimator, param_distributions=param_grid,
                                     fit_params=fit_params,
                                     # cv=3,
                                     n_iter=ngrid, scoring='neg_log_loss')  # metric" 'auc_roc','accuracy','neg_log_loss)
            gbm.fit(dTrain, yTrain)
            gbm.cv_results_
            gbm.best_estimator_
            gbm.best_score_
            gbm.return_train_score
            # gbm.feature_importance_
            Logger.debug('cv results: {}'.format(gbm.cv_results_))
            Logger.debug('best estimator: {}'.format(gbm.best_estimator_))
            Logger.debug('train_score: {}'.format(gbm.return_train_score))
            Logger.debug('best score: {}'.format(gbm.best_score_))

            ypred = gbm.predict(dTest)  # , num_iteration=gbm.best_iteration)
            # converting probabilities into 0 or 1
            predBnry = [1 if x > 0.5 else 0 for x in ypred]
            accuracy = accuracy_score(yTest, predBnry)
            accuracy
            Logger.info("Accuracy : %.2f%%" % (accuracy * 100.0))


def optimize(
             #trials,
             random_state=SEED):
    """
    This is the optimization function that given a space (space here) of
    hyperparameters and a scoring function (score here), finds the best hyperparameters.
    """
    # To learn more about XGBoost parameters, head to this page:
    # https://github.com/dmlc/xgboost/blob/master/doc/parameter.md
    space = {
        'n_estimators': hp.quniform('n_estimators', 100, 1000, 1),
        'eta': hp.quniform('eta', 0.025, 0.5, 0.025),
        # A problem with max_depth casted to float instead of int with
        # the hp.quniform method.
        'max_depth':  hp.choice('max_depth', np.arange(1, 14, dtype=int)),
        'min_child_weight': hp.quniform('min_child_weight', 1, 6, 1),
        'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
        'gamma': hp.quniform('gamma', 0.5, 1, 0.05),
        'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05),
        'eval_metric': 'auc',
        'objective': 'binary:logistic',
        # Increase this number if you have more cores. Otherwise, remove it and it will default
        # to the maxium number.
        'nthread': 4,
        'booster': 'gbtree',
        'tree_method': 'exact',
        'silent': 1,
        'seed': random_state
    }
    # Use the fmin function from Hyperopt to find the best hyperparameters
    best = fmin(score, space, algo=tpe.suggest,
                # trials=trials,
                max_evals=250)
    return best

args = parse()
initLogger()
Logger.debug('Arguments: {}'.format(args))
train_df, test_df = loadData()