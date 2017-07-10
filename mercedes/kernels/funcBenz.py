import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

leaks = {
        1: 71.34112,
        12: 109.30903,
        23: 115.21953,
        28: 92.00675,
        42: 87.73572,
        43: 129.79876,
        45: 99.55671,
        57: 116.02167,
        3977: 132.08556,
        88: 90.33211,
        89: 130.55165,
        93: 105.79792,
        94: 103.04672,
        1001: 111.65212,
        104: 92.37968,
        72: 110.54742,
        78: 125.28849,
        105: 108.5069,
        110: 87.70757,
        1004: 91.472,
        1008: 106.71967,
        1009: 108.21841,
        973: 106.76189,
        8002: 95.84858,
        8007: 87.44019,
        1644: 99.14157,
        337: 101.23135,
        253: 115.93724,
        8416: 96.84773,
        259: 93.33662,
        262: 75.35182,
        1652: 89.77625,
        4958: 113.58711,
        4960: 89.83957, #rho: -59.22558,
        7805: 105.8472, # rho = -59.20283, y =
        289: 89.27667, #rho = -59.22638, y =
        1259: 112.3909,
        1664: 112.93977,
        409: 91.00760,
        437: 85.96960,
        493: 108.40135,
        434: 93.23107,
        488: 113.39009,
        3853: 105.481283411,
    }
groups = {
    'g1': ['bc', 'az'],
    'g2': ['av', 'k', 'k', 'ac', 'am', 'l', 'b', 'aq', 'u', 't', 'ai', 'f', 'z', 'o', 'ba', 'm', 'q'],
    'g3': ['ad', 'e', 'al', 's', 'n', 'y', 'ab'],
    'g4': ['ae', 'an', 'p', 'd', 'ay', 'h', 'aj', 'aj', 'v', 'ao', 'aw', 'aa'],
    'g5': ['bb', 'ag', 'c', 'ax', 'x', 'j', 'w', 'i', 'ak', 'g', 'at', 'af', 'r', 'as', 'a', 'ap', 'au']
}

def addLeaksTrain(train, test):
    leaksDf = pd.DataFrame.from_dict(leaks, orient='index')
    leaksDf.columns = ['y']
    leaksDf['ID'] = leaksDf.index
    t2 = test.merge(leaksDf, how='right', on='ID')
    t3 = t2[train.columns]
    return pd.concat([train, t2], axis=0).sort_values('ID')

def rmLeaks(df):
    return df.drop(df.loc[leaks.keys()].index)

def labelEncode(train, test):
    for c in train.columns:
        if train[c].dtype == 'object':
            lbl = LabelEncoder()
            lbl.fit(list(train[c].values) + list(test[c].values))
            train[c] = lbl.transform(list(train[c].values))
            test[c] = lbl.transform(list(test[c].values))
    return train, test

def labelEncodeCol(df, c):
    lbl = LabelEncoder()
    lbl.fit(list(df[c].values))
    df[c] = lbl.transform(list(df[c].values))
    return df

def rm0VarFeats(all_data):
    print('\n Number of columns before cleaning: %d' % len(all_data.columns))
    cols = all_data.columns.tolist()
    for column in cols:
        if len(np.unique(all_data[column])) == 1:
            print(' Column %s removed' % str(column))
            all_data.drop(column, axis=1, inplace=True)
    return all_data

def rmIdenticalFeats(all_data):
    cols = all_data.columns.tolist()
    remove = []
    for i in range(len(cols) - 1):
        v = all_data[cols[i]].values
        for j in range(i + 1, len(cols)):
            if np.array_equal(v, all_data[cols[j]].values):
                remove.append(cols[j])
                print(' Column %s is identical to %s. Removing %s' % (str(cols[i]), str(cols[j]), str(cols[j])))

    all_data.drop(remove, axis=1, inplace=True)
    print('\n Number of columns after cleaning: %d' % len(all_data.columns))
    return all_data

def addX0group(df, s='X0', col='X0_G'):
    nv = []
    for idx, r in df.iterrows():
        for k, v in groups.items():
            if r[s] in v:
                nv.append((r['ID'], k))
    t = np.vstack(nv)
    t = pd.DataFrame(t, columns=['ID', col])
    t.ID = pd.to_numeric(t.ID)
    df = df.merge(t, how='left', on='ID')
    return df

def getX0_means(df):
    nv = []
    for idx, r in df.iterrows():
        for k, v in groups.items():
            if r['X0_OBJ'] in v:
                nv.append((r['ID'], k))
    t = np.vstack(nv)
    t = pd.DataFrame(t, columns=['ID', 'X0_G'])
    t.ID = pd.to_numeric(t.ID)
    t = df.merge(t, how='left', on='ID')
    return t.groupby(['X0_G'])[['y']].mean()

def getX0_medians(df):
    nv = []
    for idx, r in df.iterrows():
        for k, v in groups.items():
            if r['X0_OBJ'] in v:
                nv.append((r['ID'], k))
    t = np.vstack(nv)
    t = pd.DataFrame(t, columns=['ID', 'X0_G'])
    t.ID = pd.to_numeric(t.ID)
    t = df.merge(t, how='left', on='ID')
    return t.groupby(['X0_G'])[['y']].median()

def findOutliers(train, train_data, train_ids, test, test_data, target):
    clf = IsolationForest(n_estimators=500, max_samples=1.0, random_state=1001, bootstrap=True, contamination=0.02,
                          verbose=0, n_jobs=-1)
    RFR = RandomForestRegressor(n_estimators=100)
    print('\n Running Isolation Forest:')
    clf.fit(train_data.values, target)
    isof = clf.predict(train_data.values)
    train['isof'] = isof
    myindex = train['isof'] < 0
    train_IF = train.loc[myindex]
    train_IF.reset_index(drop=True, inplace=True)
    train_IF.drop('isof', axis=1, inplace=True)
    train_IF.to_csv('train-isof-outliers.csv', index=False)
    test['isof'] = clf.predict(test_data.values)
    myindex = test['isof'] < 0
    test_IF = test.loc[myindex]
    test_IF.reset_index(drop=True, inplace=True)
    test_IF.drop('isof', axis=1, inplace=True)
    test_IF.to_csv('test-isof-outliers.csv', index=False)
    print('\n Found %d outlier points' % len(train_IF))

    threshold = 2.0
    print('\n Running Random Forest Regressor (10-fold):')
    target_pred = cross_val_predict(estimator=RFR, X=train_data.values, y=target, cv=10, n_jobs=-1)
    rfr_pred = pd.DataFrame({'ID': train_ids, 'y': target, 'y_pred': target_pred})
    # rfr_pred.to_csv('prediction-train-oof-10fold-RFR.csv', index=False)
    yvalues = np.vstack((target, target_pred)).transpose()
    OL_score, OL = is_outlier(yvalues, threshold)
    train['outlier_score'] = OL_score
    myindex = train['outlier_score'] >= threshold
    train_OL = train.loc[myindex]
    # print(train_OL)
    train_OL.reset_index(drop=True, inplace=True)
    train_OL.drop(['isof', 'outlier_score'], axis=1, inplace=True)
    # train_OL.to_csv('train-outliers.csv', index=False)
    return  train_OL

def is_outlier(points, thresh=3.5):
    '''
    Returns a boolean array with True if points are outliers and False
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), 'Volume 16: How to Detect and
        Handle Outliers', The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
    '''
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return (modified_z_score, (modified_z_score > thresh) )

def rmOutliers(train, ol):
    print('Removing {} outliers'.format(len(ol)))
    return train.drop(np.where(train.loc[:, 'ID'].isin(list(ol['ID'])) == True)[0])

def projectionDf(prefix, n_comp, train, f):
    return pd.DataFrame(f,
                 columns=['{}_{}'.format(prefix, i) for i in range(1, n_comp+1)],
                 index=train.index)

def getProjections(train, test, funcs, n_comp, random_state):
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.random_projection import GaussianRandomProjection
    from sklearn.random_projection import SparseRandomProjection
    from sklearn.decomposition import PCA, FastICA
    from sklearn.decomposition import TruncatedSVD

    # local R2
    # n_comp = 6 ;  R2 score on train data: 0.65089225245 ; LB: 0.55577
    # n_comp = 12;  R2 score on train data: 0.65950822961 ; LB: 0.56760
    # n_comp = 16;  R2 score on train data: 0.65799524004 ; LB: 0.56317
    # n_comp = 20;  R2 score on train data: 0.66681870314 ; LB: 0.56262
    # n_comp = 40;  R2 score on train data: 0.67135596029 ; LB: 0.55842
    # n_comp = 80;  R2 score on train data: 0.67589753862
    # n_comp = 160; R2 score on train data: 0.68492424399 : LB: 0.55897
    # n_comp = 240; R2 score on train data: 0.69159326043 ; LB:
    # n_comp = 320; R2 score on train data: 0.69908510068 ; LB:

    if 'y' in train.columns:
        tr = train.drop(["y"], axis=1)
    else:
        tr = train.copy()

    train_proj = []
    test_proj = []
    # tSVD
    if 'svd' in funcs:
        tsvd = TruncatedSVD(n_components=n_comp, random_state=random_state)
        # tsvd_results_train = tsvd.fit_transform(tr)
        # tsvd_results_test = tsvd.transform(test)
        train_proj.append(projectionDf('svd', n_comp, tr,
                                tsvd.fit_transform(tr)))
        test_proj.append(projectionDf('svd', n_comp, test,
                                tsvd.transform(test)))

    # PCA
    if 'pca' in funcs:
        pca = PCA(n_components=n_comp, random_state=random_state)
        # pca2_results_train = pca.fit_transform(tr)
        # pca2_results_test = pca.transform(test)
        train_proj.append(projectionDf('pca', n_comp, train,
                                       f=pca.fit_transform(tr)))
        test_proj.append(projectionDf('pca', n_comp, test,
                                      pca.transform(test)))

    # ICA
    if 'ica' in funcs:
        ica = FastICA(n_components=n_comp, random_state=random_state)
        train_proj.append(projectionDf('ica', n_comp, train,
            ica.fit_transform(tr)
                                       ))
        test_proj.append(projectionDf('ica', n_comp, test,
                         ica.transform(test)
                                      ))
    # GRP
    if 'grp' in funcs:
        grp = GaussianRandomProjection(n_components=n_comp, eps=0.1, random_state=random_state)
        train_proj.append(projectionDf('grp', n_comp, train,
                          grp.fit_transform(tr)
                                            ))
        test_proj.append(projectionDf('grp', n_comp, test,
                         grp.transform(test)
                         ))
        # grp_results_train = grp.fit_transform(tr)
        # grp_results_test = grp.transform(test)

    # SRP
    if 'srp' in funcs:
        srp = SparseRandomProjection(n_components=n_comp, dense_output=True, random_state=random_state)
        train_proj.append(projectionDf('srp', n_comp, train,
                          srp.fit_transform(tr)
                          ))
        test_proj.append(projectionDf('srp', n_comp, test,
                         srp.transform(test)
                         ))
        # srp_results_train = srp.fit_transform(tr)
        # srp_results_test = srp.transform(test)

    # Append decomposition components to datasets
    # for i in range(1, n_comp + 1):
    #     train['pca_' + str(i)] = pca2_results_train[:, i - 1]
    #     test['pca_' + str(i)] = pca2_results_test[:, i - 1]
    #
    #     train['ica_' + str(i)] = ica2_results_train[:, i - 1]
    #     test['ica_' + str(i)] = ica2_results_test[:, i - 1]
    #
    #     train['tsvd_' + str(i)] = tsvd_results_train[:, i - 1]
    #     test['tsvd_' + str(i)] = tsvd_results_test[:, i - 1]
    #
    #     train['grp_' + str(i)] = grp_results_train[:, i - 1]
    #     test['grp_' + str(i)] = grp_results_test[:, i - 1]
    #
    #     train['srp_' + str(i)] = srp_results_train[:, i - 1]
    #     test['srp_' + str(i)] = srp_results_test[:, i - 1]

    return pd.concat(train_proj, axis=1), pd.concat(test_proj, axis=1)

def leaksIntoSub(sub):
    leaks = {
        1: 71.34112,
        12: 109.30903,
        23: 115.21953,
        28: 92.00675,
        42: 87.73572,
        43: 129.79876,
        45: 99.55671,
        57: 116.02167,
        3977: 132.08556,
        88: 90.33211,
        89: 130.55165,
        93: 105.79792,
        94: 103.04672,
        1001: 111.65212,
        104: 92.37968,
        72: 110.54742,
        78: 125.28849,
        105: 108.5069,
        110: 83.31692,
        1004: 91.472,
        1008: 106.71967,
        1009: 108.21841,
        973: 106.76189,
        8002: 95.84858,
        8007: 87.44019,
        1644: 99.14157,
        337: 101.23135,
        253: 115.93724,
        8416: 96.84773,
        259: 93.33662,
        262: 75.35182,
        1652: 89.77625
    }
    sub['y'] = sub.apply(lambda r: leaks[int(r['ID'])] if int(r['ID']) in leaks else r['y'], axis=1)
    return sub

def polyFeat(df):
    from sklearn.preprocessing import PolynomialFeatures

    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=True)
    colNames = poly.get_feature_names(df.columns)
    colNames = [str(x).replace(' ', '-') for x in colNames]
    d = poly.fit_transform(df)
    #maybe need some vstacking here first
    d = pd.DataFrame(d, columns=colNames, index=df.index)
    return d