# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import argparse
from sklearn.base import BaseEstimator,TransformerMixin, ClassifierMixin
import xgboost as xgb
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import ElasticNetCV, LassoLarsCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline, make_union
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold
from sklearn.utils import check_array
from functools import partial
from hyperopt import STATUS_OK, Trials
from sklearn.metrics import r2_score
import os
import getpass
import datetime
from utils.utils import dotdict, Logger
from utils.utilFunc import getProjectDir
from mercedes.kernels import funcBenz as fbz
from mercedes.kernels import modelsBenz as mbz
from utils.utilFunc import pickleAway
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, space_eval

def run():

    #set file Paths
    projectDir = getProjectDir()

    #define all parameters and seach space
    params = dotdict([
        ('ex', 1),

        ('leaksToTrain', True),
        ('rm0VarFeats', True),
        ('rmIdenticalFeats', True),
        ('rmOutliers', True),
        ('engProj', True),
        ('leaksIntoSub', True),
        ('addX0groups', True),
        ('polyFeat', False),
        ('runXgbCV', False),
        ('readOL', False),
        ('seedRounds', 10),
        ('kfold', 1),
        ('max_evals', 1),

        ('saveXgbModel', 1),
        ('saveXgbFeatImp', 1),
    ])

    Logger.init_log(os.path.join(projectDir,'log/log-{}'.format(datetime.date.today())))
    parser = argparse.ArgumentParser(description='Laala')
    parser.add_argument('-srcargs', type=int, action='store', help='', default=1)
    parser.add_argument('-gpu', action='store', help='', default='grow_gpu')
    if os.name == 'posix':
        args = parser.parse_args()
        if args.srcargs == 1:
            # argsDef = ['-srcargs', '1', '-gpu', '0']
            argsDef = ['-gpu', 'grow_gpu']
            args = parser.parse_args(argsDef)
    else:
        argsDef = ['-srcargs', '1', '-gpu', '0']
        args = parser.parse_args(argsDef)

    Logger.debug('Arguments: {}'.format(args))

    # any classes here
    class StackingEstimator(BaseEstimator, TransformerMixin):

        def __init__(self, estimator):
            self.estimator = estimator

        def fit(self, X, y=None, **fit_params):
            self.estimator.fit(X, y, **fit_params)
            return self
        def transform(self, X):
            X = check_array(X)
            X_transformed = np.copy(X)
            # add class probabilities as a synthetic feature
            if issubclass(self.estimator.__class__, ClassifierMixin) and hasattr(self.estimator, 'predict_proba'):
                X_transformed = np.hstack((self.estimator.predict_proba(X), X))

            # add class prodiction as a synthetic feature
            X_transformed = np.hstack((np.reshape(self.estimator.predict(X), (-1, 1)), X_transformed))

            return X_transformed

    #get source data
    train = pd.read_csv(os.path.join(projectDir, r'input/train.csv'), dtype={'ID': np.int32})
    test = pd.read_csv(os.path.join(projectDir, r'input/test.csv'), dtype={'ID': np.int32})

    if params.addX0groups:
        train = fbz.addX0group(train)
        test = fbz.addX0group(test)

    y_train = train['y'].values
    train_ids = train['ID'].values
    train = train.drop(['ID', 'y'], axis=1)
    print('\n Initial Train Set Matrix Dimensions: %d x %d' % (train.shape[0], train.shape[1]))
    train_len = len(train)
    test_ids = test['ID'].values
    test = test.drop(['ID'], axis=1)
    print('\n Initial Test Set Matrix Dimensions: %d x %d' % (test.shape[0], test.shape[1]))

    #label encode categoricals
    train, test = fbz.labelEncode(train, test)

    all_data = pd.concat((train, test))

    # remove non-unique feats
    if params.rm0VarFeats:
        all_data = fbz.rm0VarFeats(all_data)
    # Remove identical columns where all data points have the same value
    if params.rmIdenticalFeats:
        all_data = fbz.rmIdenticalFeats(all_data)

    features = all_data.columns
    print('\n Final Matrix Dimensions: %d x %d' % (all_data.shape[0], all_data.shape[1]))
    train_data = pd.DataFrame(all_data[ : train_len].values, columns=features)
    test_data = pd.DataFrame(all_data[train_len : ].values, columns=features)
    train_data.reset_index(drop=True, inplace=True)
    test_data.reset_index(drop=True, inplace=True)

    #FEAT ENGINEERING
    all_data = pd.concat((train, test))
    # group X0 and label encode
    # review the snte again
    # switch on poly feat batch processing
    # more feat eng from all threads

    # PCA / ICA / GRP / tSVD
    #save columns list before adding the decomposition components
    usable_columns = list(set(train.columns) - set(['y']))
    if params.engProj:
        train_proj, test_proj = fbz.getProjections(train, test, ['svd','pca','ica','grp','srp'], n_comp=12, random_state=420)
        train = pd.concat([train, train_proj], axis=1)
        test = pd.concat([test, test_proj], axis=1)

    # perform parameterized outlier detection and handling
    if params.rmOutliers:
        train.insert(0, 'ID', train_ids)
        train.insert(0, 'y', y_train)
        test.insert(0, 'ID', test_ids)
        if params.readOL:
            train_OL = pd.read_csv(os.path.join(projectDir, 'model/train-outliers.csv'))
        else:
            train_OL = fbz.findOutliers(train.copy(), train_data, train_ids, test.copy(), test_data, target=y_train)

        # train.drop(['isof', 'outlier_score'], axis=1, inplace=True)
        # test.drop(['isof'], axis=1, inplace=True)
        train = fbz.rmOutliers(train, train_OL)
        #optional, drop y outliers  by Liam
        train = train[train.y < 250] # Optional: Drop y outliers
        train = train.drop(np.where((train.loc[:,'ID'] == 0)==True)[0])

    #add leaks to train data
    if params.leaksToTrain:
        # rework the function to take it from test
        train = fbz.addLeaksTrain(train, test)
        train = train.reset_index(drop=True)

    if params.polyFeat:
        dfPoly = fbz.polyFeat(all_data)
        all_data = pd.concat([all_data, dfPoly], axis=1)

    y_mean = np.mean(y_train)
    test_ids = test['ID'].values
    #finaltrainset and finaltestset are data to be used only the stacked model (does not contain PCA, SVD... arrays)
    stack_trainset = train[usable_columns].values
    stack_testset = test[usable_columns].values
    stack_y_train = train['y']

    # TRAIN 1 for Stacking
    # Models - with guessed params ( optimize when hyperopt params come in)
    # include lgbm

    # somewhere get in that recursive feature selection
    # train -> save Imps -> pick all relevant or as many as fit into memory / alternatively batch model training (new)
    # or combo of abvoe
    # use sklearn pipeline or automated feat selection packages for that

    # TRAIN 2 for Prediction
        #hyperopt wrapper to optimize each model separately on 5 fold local CV results, finally use only 1 space setting

    #SEED WRAPPER


    '''5. Group median and perhaps other simple statistical methods'''
    test_y_id = mbz.predWithX0(train, df=test, how='mean')
    train_y_id = mbz.predWithX0(train, df=train.copy(), how='mean')
    test_y_id.columns = ['ID','Y_X0']
    train_y_id.columns = ['ID', 'Y_X0']
    test = test.merge(test_y_id.loc[:, ['ID', 'Y_X0']], how='left', on='ID')
    train = train.merge(train_y_id.loc[:, ['ID', 'Y_X0']], how='left', on='ID')
    xgbPredsTest = []
    xgbPredsTrain = []

    for seedRound in range(0, params.seedRounds):
        np.random.seed(seedRound)
        '''1. XGB Model & predict'''
        # best_p= {
        # 'n_trees': [500, 600, 650, 650, 600, 700],
        # 'colsample_bytree': [0.9, 0.9, 0.86, 0.98, 0.94],
        # 'colsample_bylevel': [0.8, 0.96, 1, 0.92, 1],
        # 'min_child_weight': [2, 3, 1, 3, 1],
        # 'max_depth': 3,
        # 'gamma': [0, 0.18, 0.02, 0.14, 0.18],
        # 'subsample': [1,0.9,0.96,0.88,1],
        # 'iter': [1175, 869, 932, 839, 1527],
        # 'iter2': [962, 1082, 744, 1358, 835, 938, 999, 965, 870, 1017]
        # }
        # for k, v in best_p.items():
        #     print(np.mean(v))
        # 616.666666667
        # 0.916
        # 0.936
        # 2.0
        # 3.0
        # 0.104
        # 0.948

        xgbSpace = {
            'learning_rate': 0.0045,  # hp.quniform('learning_rate', 0.01, 0.2, 0.01), alias: eta
            # A problem with max_depth casted to float instead of int with
            # the hp.quniform method.
            'max_depth': 3, #hp.choice('max_depth', np.arange(3, 6, dtype=int)), #4,
            'min_child_weight': 2,#hp.quniform('min_child_weight', 1, 6, 1),
            'subsample': 0.95, #hp.quniform('subsample', 0.9, 1, 0.02), #
            'n_trees': 620,
            'gamma': 0.1, #hp.quniform('gamma', 0, 0.2, 0.02),
            'colsample_bytree': 0.92, # hp.quniform('colsample_bytree', 0.8, 1, 0.02),
            'colsample_bylevel': 0.94, #hp.quniform('colsample_bylevel', 0.8, 1, 0.02),
            # 'max_delta_step': xgbparams.max_delta_step,  # 0,
            # colsample_bylevel = 1,
            # reg_alpha = 0,
            # reg_lambda = 1
            # 'scale_pos_weight': xgbparams.scale_pos_weight,
            'objective': 'reg:linear',
            'eval_metric': 'rmse',
            'base_score': y_mean,  # base prediction = mean(y_train)
            'silent': True,
            # 'booster': xgbparams.booster,
            'tree_method': 'exact',
            'seed': seedRound*10,
            # 'missing': None,
            'xgbArgs': {
                'num_boost_round': 977,#745,
                # sample(scope.int(hp.quniform('num_boost_round', 100, 1000, 1))),
                'verbose_eval': 50,
            }}
        if args.gpu != '0':
            xgbSpace['updater'] = args.gpu

        hyperOptTrials = Trials()
        params.batch = seedRound
        #replace this with a 5-fold CV if parameters are better known and all is tested for long runs
        X = train.drop(['ID','y'], axis=1)#.values
        y = train['y']#.values
        testxgb = test[train.drop('y', axis=1).columns]
        testxgb = testxgb.drop('ID', axis=1)

        ss = ShuffleSplit(n_splits=2, test_size=0.2, random_state=seedRound)

        # if params.runXgbCV:
        #     kFold = KFold(n_splits=params.kfold, shuffle=True, random_state=seedRound)
        #     i=0
        #     for train_index, test_index in kFold.split(X):
        #
        #     # for train_index, test_index in ss.split(X):
        #         print("TRAIN:", len(train_index), "VALIDATION:", len(test_index))
        #         x_train, x_valid = X.iloc[train_index,:], X.iloc[test_index,:]
        #         y_train, y_valid = y.iloc[train_index], y.iloc[test_index]
        # else:
        x_train = X
        y_train = y

        xgbMTrain = xgb.DMatrix(x_train, label=y_train, feature_names=x_train.columns)
        xgbMTest = xgb.DMatrix(testxgb, feature_names=testxgb.columns)
        # if params.runXgbCV:  # run a CV
        #     xgbMValid = xgb.DMatrix(x_valid, label=y_valid, feature_names=x_valid.columns)
        #     # specify validations set to watch performance
        #     xgbSpace['xgbArgs']['evals'] = [(xgbMTrain, 'train'), (xgbMValid, 'val')]
        #     xgbSpace['xgbArgs']['evals_result'] = {}
        #     xgbSpace['xgbArgs']['early_stopping_rounds'] = 50

        xgbFunc = partial(mbz.xgbTrain, xgbMTrain=xgbMTrain, params=params)
        paramsx = params
        # paramsx['max_evals']=1
        best_params = mbz.optimize(space=xgbSpace, scoreF=xgbFunc, trials=hyperOptTrials, params=paramsx)
        xgbmodel = hyperOptTrials.best_trial['result']['model']
        # pickleAway(xgbmodel, ex='ex{}'.format(params.ex), fileNStart='xgbModel', dir1=projectDir, dir2='model',
        #            batch=params.batch)
        # Logger.info('Fold: {} - best hyperopt params: {}'.format( best_params))

        # if params.runXgbCV:
        #     bestIter = hyperOptTrials.best_trial['result']['bestIter']
        #     Logger.info('XGB CV evals result: {}'.format(hyperOptTrials.best_trial['result']['evals_result']))
        #     Logger.info('XGB CV bestIter: {}'.format(hyperOptTrials.best_trial['result']['bestIter']))

        xgbPredsTest.append( xgbmodel.predict(xgbMTest) )
        xgbPredsTrain.append( xgbmodel.predict(xgbMTrain) )
        r2Train = r2_score(y_train, xgbPredsTrain[-1])
        Logger.info('xgb R2 Train - {}, seed-{}, fold-{}'.format(r2Train, seedRound, 0))

        # pickleAway(hyperOptTrials, ex='ex{}'.format(params.ex), fileNStart='xgbHyperOptTrial', dir1=projectDir, dir2='hyperOptTrials',
        #            batch=0)
    xgbPredTest = np.sum(xgbPredsTest, axis=0) / ( params.kfold *  params.seedRounds )
    xgbPredTrain = np.sum(xgbPredsTrain, axis=0) / ( params.kfold * params.seedRounds )
    subTrain = pd.DataFrame()
    subTest = pd.DataFrame()
    subTest['y'] = xgbPredTest
    subTrain['y'] = xgbPredTrain
    subTest['ID'] = test_ids
    subTest.to_csv(os.path.join(projectDir, r'subm/myBenzTestirstxgbModel-{}.csv'.format(
        datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))), index=False)
    # subTrain.to_csv(os.path.join(projectDir, r'subm/myBenzTrainFirstxgbModel-{}.csv'.format(
    #     datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))), index=False)
    #Stack its predictions
    train['y_XGB'] = xgbPredTrain
    test['y_XGB'] = xgbPredTest

    stTrainPreds = []
    stTestPreds = []
    for seedRound in range(0, params.seedRounds):
        np.random.seed(seedRound)

        '''2. Train stacked models & predict the test data'''

        stSpace = {
            'learning_rate': 0.0001,  # hp.quniform('learning_rate', 0.01, 0.2, 0.01), alias: eta
            'loss': 'huber',
            'max_depth': 3, #hp.choice('max_depth', np.arange(2, 5, dtype=int)), #3,
            'max_features': 0.55, #hp.quniform('max_features', 0.4, 0.7, 0.05), # 0.55,
            'min_samples_leaf': 18, # hp.choice('min_samples_leaf', np.arange(10, 26, 2)), #18
            'min_samples_split': 14, #hp.choice('min_samples_split', np.arange(8, 20, 2)), #14,
            'subsample': 0.7# hp.quniform('subsample', 0.6, 0.9, 0.01), #0.7
        }

        def stackedPred(stSpace, stack_trainset, stack_y_train):
            stacked_pipeline = make_pipeline(
                StackingEstimator(estimator=LassoLarsCV(normalize=True)),
                StackingEstimator(estimator=GradientBoostingRegressor(
                    **stSpace)),
                LassoLarsCV()
            )
            stacked_pipeline.fit(stack_trainset, stack_y_train)
            # skStackPred = stacked_pipeline.predict(stack_testset)
            skStackPredTrain = stacked_pipeline.predict(stack_trainset)
            Logger.info('During optimization: R2 on train data: {}'.format(r2_score(stack_y_train, skStackPredTrain)))
            return {'loss': r2_score(stack_y_train, skStackPredTrain),
                'status': STATUS_OK,
                'model': stacked_pipeline
                }

        print(params)
        params['max_evals']=1
        stHyperOptTrials = Trials()
        stackFunc = partial(stackedPred, stack_trainset=stack_trainset, stack_y_train=stack_y_train)
        best_params = mbz.optimize(space=stSpace, scoreF=stackFunc, trials=stHyperOptTrials, params=params)
        Logger.info('Best stack hyperopt params: {}'.format(best_params))
        st_model = stHyperOptTrials.best_trial['result']['model']
        # pickleAway(stHyperOptTrials, ex='ex{}'.format(params.ex), fileNStart='stHyperOptTrial', dir1=projectDir,
        #            dir2='hyperOptTrials', batch=0)
        stTrainPreds.append(st_model.predict(stack_trainset))
        stTestPreds.append(st_model.predict(stack_testset))

    skStackPredTest = np.sum(stTestPreds, axis=0) / params.seedRounds
    skStackPredTrain = np.sum(stTrainPreds, axis=0) / params.seedRounds
    Logger.info('Best Stacked model: R2 on train data: {}'.format(r2_score(stack_y_train, skStackPredTrain)))

    sub = pd.DataFrame()
    sub['y'] = skStackPredTest
    sub['ID'] = test_ids
    sub.to_csv(os.path.join(projectDir, r'subm/myBenzTestStackerModel-{}.csv'.format(
        datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))), index=False)


    train['y_SK'] = skStackPredTrain
    test['y_SK'] = skStackPredTest

    '''3. Neural networks model'''
    if False:
        nnSpace = dotdict([
            #####################
            # Neural Network
            #####################
            # Training steps
            ('STEPS', 500),
            ('LEARNING_RATE', 1),# 0.0001),
            ('BETA', 0.01),
            ('DROPOUT', 0.5),
            ('RANDOM_SEED', 12345),
            ('MAX_Y', 250),
            ('RESTORE', True),
            ('START', 0),
            # Training variables
            ('IN_DIM', 13),
            # Network Parameters - Hidden layers
            ('n_hidden_1', 100),
            ('n_hidden_2', 50),
        ])
        nnHyperOptTrials = Trials()
        nnFunc = partial(mbz.deepNN, train=train, y_train=train['y'])
        nnBest_params = mbz.optimize(space=nnSpace, scoreF=nnFunc, trials=nnHyperOptTrials, params=params)
        nnModel = nnHyperOptTrials.best_trial['result']['model']

        #Convert to matrix
        test_submit = test.as_matrix()
        test_submit_id = test_ids

    '''4. TPOT automated ML approach'''
    from tpot import TPOTRegressor
    test_ids = test['ID']
    train = train.drop('ID', axis=1)
    auto_classifier = TPOTRegressor(generations=30, population_size=15, verbosity=2)
    from sklearn.model_selection import train_test_split

    X_train, X_valid, y_train, y_valid = train_test_split(train.drop('y', axis=1), train['y'],
                                                          train_size=0.75, test_size=0.25)
    auto_classifier.fit(X_train, y_train)
    # pickleAway(auto_classifier, ex='ex{}'.format(params.ex), fileNStart='tpotModel', dir1=projectDir, dir2='model',
    #            batch=0)
    Logger.info('The cross-validation MSE: {}'.format(auto_classifier.score(X_valid, y_valid)))
    # auto_classifier.export(os.path.join(projectDir, r'model/tpotClassifier'))

    # we need access to the pipeline to get the probabilities
    y_tpot = auto_classifier.predict(test.drop('ID', axis=1))
    y_train_tpot = auto_classifier.predict(train.drop('y', axis=1))
    Logger.info('Tpot R2 on train data: {}'.format(r2_score(train['y'], y_train_tpot)))
    if True:
        sub_tpot = pd.DataFrame()
        sub_tpot['ID'] = test_ids
        sub_tpot['y'] = y_tpot
        sub_tpot.to_csv(os.path.join(projectDir, r'subm/myBenzTpotModel-{}.csv'.format(
            datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))), index=False)

    '''6. LGBM'''
    if False:
        import lightgbm as lgb

        ss = ShuffleSplit(n_splits=2, test_size=0.2, random_state=params.seedRounds)
        for train_index, test_index in ss.split(X):
            print("TRAIN:", len(train_index), "VALIDATION:", len(test_index))
            x_train, x_valid = X.iloc[train_index, :], X.iloc[test_index, :]
            y_train, y_valid = y.iloc[train_index], y.iloc[test_index]

        lgbSpace = {
            'learning_rate': 0.0045,  # hp.quniform('learning_rate', 0.01, 0.2, 0.01), alias: eta
            'max_depth': 4,  # hp.choice('max_depth', np.arange(3, 10, dtype=int)),
            'subsample': 0.93,  # hp.quniform('subsample', 0.5, 1, 0.1),
            'n_trees': 520,
            'objective': 'reg:linear',
            'metric': 'rmse',
            'base_score': y_mean,  # base prediction = mean(y_train)
            'silent': True,
            # 'booster': xgbparams.booster,
            'tree_method': 'exact',
            'seed': seedRound,
            # 'missing': None,
            'xgbArgs': {
                'num_iteration': 1250,
                # sample(scope.int(hp.quniform('num_boost_round', 100, 1000, 1))),
                'verbose_eval': 50,
            }}

        lgbTrain = lgb.Dataset(x_train, label=y_train)
        lgbValid = lgb.Dataset(x_valid, label=y_valid, reference=train_data)
        lgbTest = lgb.Dataset(test)

        valid_sets = [train_data, test_data]

        lgbFunc = partial(mbz.lgbTrain, args=args, xgbMTrain=lgbTrain, xgbMTest=lgbValid, params=params)
        best_params = mbz.optimize(space=lgbSpace, scoreF=lgbFunc, trials=hyperOptTrials, params=params)
        bestIter = hyperOptTrials.best_trial['result']['bestIter']
        model = hyperOptTrials.best_trial['result']['model']
        lgbPred = model.predict(lgbTest)

        ypred = lgbm.predict(xTest, num_iteration=lgbm.best_iteration)

        # converting probabilities into 0 or 1
        predBnry = [1 if x > 0.5 else 0 for x in ypred]
        accuracy = accuracy_score(yTest, predBnry)
        Logger.info("Accuracy : %.2f%%" % (accuracy * 100.0))

        estimator = lgb.LGBMClassifier(nthread=-1,
                                       silent=False)  # ,categorical_feature=[list(X_train).index(catFeature) for catFeature in categorical_features])
        fit_params = None
        fit_params = {
            # 'num_round': param_grid['num_round'],
            'eval_set': [(dTrain, yTrain), (dTest, yTest)],
            # 'eval_metric': 'binary_log_loss',
            'early_stopping_rounds': 5
        }
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
    # note that talibs are generated with function A
    # dotdict(function, function, category)


            # just list feats that can be engineered. a dic, featName, function that gens it, inp, out
            # kind copy talibs design for this


        '''R2 Score on the entire Train data when averaging'''

        print('R2 score on train data:')
        # r2_score(train['y'], y)
        print(r2_score(y_train,stacked_pipeline.predict(finaltrainset)*0.2855 + model.predict(xgbMTrain)*0.7145))

        '''Average the preditionon test data  of both models then save it on a csv file'''


        # ENSEMBLE all weighted by public leaderboard score
        # or first majority vote, then weight of local CV
        sub = pd.DataFrame()
        sub['y'] += xgbPred*0.75 + skStackPred*0.25


    # ENSMBLE for each model
    xgb = pd.read_csv(os.path.join(projectDir, r'subm/final/'), index=False)
    stack = pd.read_csv(os.path.join(projectDir, r'subm/final/'), index=False)
    tpot = pd.read_csv(os.path.join(projectDir, r'subm/final/'), index=False)

    sub['y'] += xgb * 0.75 + stack * 0.25

    if params.leaksIntoSub:
        sub = fbz.leaksIntoSub(sub)

    sub.to_csv(os.path.join(projectDir,r'subm/myBenz-{}.csv'.format(
                datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))), index=False)

if __name__ == '__main__':
    run()