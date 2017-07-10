import csv
import os
import errno
import time
import datetime
import pandas as pd
import numpy as np
import pickle
import getpass

def saveTestResult(filename=None, folder=r'../log', **kwargs):

    if filename is None:
        st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
        filename = st

    mode='a' if os.path.isfile(os.path.join(folder, filename)) else 'w'

    with open(os.path.join(folder, filename), mode=mode) as f:
        w = csv.DictWriter(f, kwargs.keys())
        w.writeheader()
        w.writerow(kwargs)

def create_feature_map(features, projectDir):
    outfile = open(os.path.join(projectDir, 'model/xgb.fmap'), 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()

def pickleAway(obj, dir1, dir2=None, ex=None, fileNStart='tmp', batch=0):
    dir = dir1 + '/' + dir2 if dir2 is not None else dir1
    if ex is not None:
        dir = os.path.join(dir, ex)
        createDir(dir)
    pickle.dump(obj, open(
        os.path.join(dir,'{}-{}-{}batch-{}'.format(
            fileNStart,
            getpass.getuser(),
            batch,
            datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        )), 'wb'))

def logBenchmark(Logger,yTrain,yTest):
    Logger.debug('Benchmark alwUp Train / Test: {} / {}'.format(
        sum(yTrain) / len(yTrain),
        sum(yTest) / len(yTest)
    ))

def getImportantFeatures(cumsumThreshold, dir1, dir2, required=['close']):
    folder = os.path.join(dir1, dir2)
    featImpList = []
    for root, dirs, filenames in os.walk(folder):
        for file in filenames:
            featImpList.append(pickle.load(open(os.path.join(folder, file), 'rb')))
    featImpListMerged = []
    sel=[]
    for feats in featImpList:
        sel.append(feats.iloc[np.where(feats['fscore'].cumsum() < cumsumThreshold)[0].tolist(), :])
        featImpListMerged = featImpListMerged + feats['feature'].tolist()
    featImpListMerged = list(set(featImpListMerged))
    sel = pd.concat(sel, axis=0)['feature'].tolist()
    sel = list(set(sel + required))
    return sel, [x for x in featImpListMerged if x not in sel]

def createDir(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def interpolateTS(df, maxmin=10, cols=['open','high','low','close']):
    slices, leave = getNullRanges(df, maxmin)
    for s in slices:
        df.loc[s[0]:s[1], cols] = df.loc[s[0]:s[1],cols].interpolate(method='linear')
    return df, leave

def getNullRanges(df, maxmin=10):
    interp = []
    leave = []
    inb = False
    for i in range(0, len(df)):
        if pd.isnull(df.iloc[i, 3]):
            if inb == False:
                start = df.index[i-1]
                startN = df.index[i]
            inb = True
        else:
            if inb == True:
                end = df.index[i]
                endN = df.index[i-1]
                if end-start <= datetime.timedelta(minutes = maxmin):
                    interp.append((start, end))
                else:
                    leave.append((startN, endN))
            inb = False
    return interp, leave

def getProjectDir():
    if os.name == 'posix':
        projectDir = r'/home/' + getpass.getuser() + r'/repos/kaggle/mercedes'
    else:
        projectDir = r'C:\repos\kaggle\mercedes'
    return projectDir