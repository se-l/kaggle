import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, space_eval
import xgboost as xgb
from utils.utilFunc import pickleAway, getProjectDir
import operator
import os, datetime
from utils.utils import dotdict, Logger
from sklearn.metrics import r2_score
import tensorflow as tf
from sklearn.model_selection import KFold

projectDir = getProjectDir()
Logger.init_log(os.path.join(projectDir,'log/log2-{}'.format(datetime.date.today())))

def optimize(scoreF, space, trials, params):
    """
    This is the optimization function that given a space (space here) of
    hyperparameters and a scoring function (score here), finds the best hyperparameters.
    """
    # To learn more about XGBoost parameters, head to this page:
    # https://github.com/dmlc/xgboost/blob/master/doc/parameter.md

    # Use the fmin function from Hyperopt to find the best hyperparameters
    return space_eval(space,
                      fmin(scoreF,
                           space,
                           algo=tpe.suggest,
                           trials=trials,
                           verbose=1,
                           max_evals=params.max_evals)
                      )

def xgbTrain(xgbparams, args, xgbMTrain, xgbMTest, params, cv=False):

    # Logger.info('xgbparams: {}'.format(xgbparams))
    xgbArgs = xgbparams['xgbArgs']
    del xgbparams['xgbArgs']
    # Logger.info('Train Matrix row / col: {} - {}'.format(xgbMTrain.num_row, xgbMTrain.num_col))
    ############    TRAIN   ############################
    model = xgb.train(xgbparams, xgbMTrain,
                      **xgbArgs,
                      )
    if args.saveLog != 0:
        pickleAway(model,ex='ex{}'.format(args.ex) ,fileNStart='xgbModel',dir1=projectDir,dir2='model',batch=params.batch)

    featImp = pd.DataFrame(
        sorted(model.get_fscore().items(), key=operator.itemgetter(1), reverse=True),
        columns=['feature', 'fscore'])
    featImp['fscore'] = featImp['fscore'] / featImp['fscore'].sum()
    Logger.info('Top 3 Feats:\n {}'.format(featImp.sort_values(by=['fscore'], ascending=False)[:3]))
    if args.saveLog != 0:
        pickleAway(featImp,ex='ex{}'.format(args.ex),fileNStart='xgbFImp',dir1=projectDir,dir2='FImp',batch=params.batch)

    return {'loss': model.best_score,
            'status': STATUS_OK,
            'bestIter': model.best_iteration,
            'fscore': model.get_fscore(),
            'model': model
            }

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.03, shape=shape)
    return tf.Variable(initial)

def deep_network(inputs, keep_prob, nnSpace):
    # Input -> Hidden Layer
    w1 = weight_variable([nnSpace.IN_DIM, nnSpace.n_hidden_1])
    b1 = bias_variable([nnSpace.n_hidden_1])
    # Hidden Layer -> Hidden Layer
    w2 = weight_variable([nnSpace.n_hidden_1, nnSpace.n_hidden_2])
    b2 = bias_variable([nnSpace.n_hidden_2])
    # Hidden Layer -> Output
    w3 = weight_variable([nnSpace.n_hidden_2, 1])
    b3 = bias_variable([1])

    # 1st Hidden layer with dropout
    h1 = tf.nn.relu(tf.matmul(inputs, w1) + b1)
    h1_dropout = tf.nn.dropout(h1, keep_prob)
    # 2nd Hidden layer with dropout
    h2 = tf.nn.relu(tf.matmul(h1_dropout, w2) + b2)
    h2_dropout = tf.nn.dropout(h2, keep_prob)

    # Run sigmoid on output to get 0 to 1
    out = tf.nn.sigmoid(tf.matmul(h2_dropout, w3) + b3)

    # Loss function with L2 Regularization
    regularizers = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(w3)

    scaled_out = tf.multiply(out, nnSpace.MAX_Y)  # Scale output
    return inputs, out, scaled_out, regularizers

def deepNN(nnSpace, train, y_train):
    #########################
    # Create data
    #########################
    features = ['X0',
                'X5',
                'X118',
                'X127',
                'X47',
                'X315',
                'X311',
                'X179',
                'X314',
                'X232',
                'X29',
                'X263',
                'X261']
    train = train.loc[:, features]

    tf.set_random_seed(nnSpace.RANDOM_SEED)

    # Create the model
    x = tf.placeholder(tf.float32, [None, nnSpace.IN_DIM])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 1])

    # Dropout on hidden layers
    keep_prob = tf.placeholder("float")

    # Build the graph for the deep net
    inputs, out, scaled_out, regularizers = deep_network(x, keep_prob, nnSpace)

    # Normal loss function (RMSE)
    loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_, scaled_out))))

    # Loss function with L2 Regularization
    loss = tf.reduce_mean(loss + nnSpace.BETA * regularizers)

    # Optimizer
    train_step = tf.train.AdamOptimizer(nnSpace.LEARNING_RATE).minimize(loss)

    total_error = tf.reduce_sum(tf.square(tf.subtract(y_, tf.reduce_mean(y_))))
    unexplained_error = tf.reduce_sum(tf.square(tf.subtract(y_, scaled_out)))
    accuracy = tf.subtract(1.0, tf.divide(unexplained_error, total_error))

    # Save model
    saver = tf.train.Saver(max_to_keep=5)

    with tf.Session() as sess:
        # if RESTORE:
        #    print('Loading Model...')
        #    ckpt = tf.train.get_checkpoint_state('./models/neural/')
        #    saver.restore(sess, ckpt.model_checkpoint_path)
        # else:
        sess.run(tf.global_variables_initializer())

        # train, y_train, test_submit, test_submit_id = get_data()

        # Train until maximum steps reached or interrupted
        for i in range(nnSpace.START, nnSpace.STEPS):
            k_fold = KFold(n_splits=5, shuffle=True)
            # if i % 100 == 0:
            #    saver.save(sess, './models/neural/step_' + str(i) + '.cptk')

            for k, (ktrain, ktest) in enumerate(k_fold.split(train, y_train)):
                train_step.run(feed_dict={x: train[ktrain], y_: y_train[ktrain], keep_prob: nnSpace.DROPOUT})
                # Show test score every 10 iterations
                if i % 10 == 0:
                    # Tensorflow R2
                    # train_accuracy = accuracy.eval(feed_dict={
                    #    x: train[ktest], y_: y_train[ktest]})
                    # SkLearn metrics R2
                    train_accuracy = r2_score(y_train[ktest],
                                              sess.run(scaled_out, feed_dict={x: train[ktest], keep_prob: 1.0}))
                    print('Step: %d, Fold: %d, R2 Score: %g' % (i, k, train_accuracy))

        ####################
        # CV (repeat 5 times)
        ####################
        CV = []
        for i in range(5):
            k_fold = KFold(n_splits=10, shuffle=True)
            for k, (ktrain, ktest) in enumerate(k_fold.split(train, y_train)):
                # Tensorflow R2
                # accuracy = accuracy.eval(feed_dict={
                #    x: train[ktest], y_: y_train[ktest]})
                # SkLearn metrics R2
                accuracy = r2_score(y_train[ktest],
                                    sess.run(scaled_out, feed_dict={x: train[ktest], keep_prob: 1.0}))
                print('Step: %d, Fold: %d, R2 Score: %g' % (i, k, accuracy))
                CV.append(accuracy)
        print('Mean R2: %g' % (np.mean(CV)))

        return {'loss': np.mean(CV),
                'status': STATUS_OK,
                # 'bestIter': model.best_iteration,
                # 'fscore': model.get_fscore(),
                'model': saver
                }

def xgbCV(xgbparams, args, xgbMTrain, xgbMTest, params):

    xgbArgs = xgbparams['xgbArgs']
    del xgbparams['xgbArgs']
    evals_result = {}

    cvresult = xgb.cv(xgbparams, xgbMTrain,
                      **xgbArgs,
                      nfold=params.nfold,
                      stratified=True,
                      show_stdv=False)

    return {'loss': cvresult.iloc[-1, 0],
            'status': STATUS_OK,
            'bestIter': len(cvresult)
            }
