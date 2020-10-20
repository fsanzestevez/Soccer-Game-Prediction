# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 12:26:28 2020

@author: Fran
"""

from dataManipulation import DataManipulation

from sklearn.model_selection import TimeSeriesSplit
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import ElasticNet, SGDClassifier
from sklearn.metrics import roc_curve, auc
import numpy as np
import pandas as pd
import math
import itertools
import random
import pickle
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle


class Modelling():
    def __init__(self, max_iter=100):
        self.max_iter = max_iter
        self.n_classes = 3
    
    def train_Models(self, train):
        model_params = dict()
        # Since we can't do CV in time-series, we apply our own GridSearch
        tscv = TimeSeriesSplit(n_splits=5)
        i = 1
        for tr_index, val_index in tscv.split(train):
            train_chunk = train.iloc[tr_index, :]
            val_chunk = train.iloc[val_index, :]
            manipulate = DataManipulation(train_chunk, val_chunk)
            
            X_tr = manipulate.x_train.copy()
            X_val = manipulate.x_test.copy()

            y_tr = manipulate.y_train.copy()
            y_val = manipulate.y_test.copy()
            
            params_rf = self.train_RF(X_tr, y_tr, X_val, y_val, i)
            model_params['RF_'+str(i)] = params_rf
            params_gbm = self.train_GBM(X_tr, y_tr, X_val, y_val, i)
            model_params['GBM_'+str(i)] = params_gbm
            params_en = self.train_EN(X_tr, y_tr, X_val, y_val, i)
            model_params['EN_'+str(i)] = params_en

            i += 1
        
        self.model_params = model_params
        return
            
    def train_RF(self, X_tr, y_tr, X_val, y_val, split_no=0):
        max_iter = self.max_iter
        i = 0
        max_score = 00
            
        n_feat = X_tr.shape[1]
        
        
        
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start = 50, stop = 500,
                                                    num = 20)]
        # Number of features to consider at every split
        max_features = [int(x) for x in np.linspace(int(math.sqrt(n_feat)/2),
                                                    n_feat, 10)]
        
        
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4, 10]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        
        params = list(itertools.product(n_estimators, max_features,
                                       min_samples_split, min_samples_leaf,
                                       bootstrap))
        param_df = pd.DataFrame(columns=['iteration', 'n_estimators', 
                                         'max_features', 'min_samples_split',
                                         'min_samples_leaf', 'bootstrap',
                                         'score'],
                                index=range(len(params)))
        r = list(range(len(params)))
        random.shuffle(r)
        for random_i in r:
            if i > max_iter:
                break

            params_i = params[random_i]
            rfc = RandomForestClassifier(
                n_estimators=params_i[0],
                max_features=params_i[1],
                min_samples_split=params_i[2],
                min_samples_leaf=params_i[3],
                bootstrap=params_i[4],
                n_jobs=-1)
            
            clf = OneVsRestClassifier(rfc)
            clf.fit(X_tr, y_tr)
            score_i = clf.score(X_val, y_val)
            if score_i > max_score:
                max_score = score_i
                
                
            param_df['iteration'][i] = i
            param_df['n_estimators'][i] = params_i[0]
            param_df['max_features'][i] = params_i[1]
            param_df['min_samples_split'][i] = params_i[2]
            param_df['min_samples_leaf'][i] = params_i[3]
            param_df['bootstrap'][i] = params_i[4]
            param_df['score'][i] = score_i
            i += 1
        
        return param_df
    
    def train_GBM(self, X_tr, y_tr, X_val, y_val, split_no=0):
        max_iter = self.max_iter
        i = 0
        max_score = 0
        best_i = 0
       
        # learning rate
        learning_rate = [x for x in np.linspace(0.05, 0.5, 10)]
        
        min_samples_split = [x for x in np.linspace(0.005, 0.01, 5)]
        
        # Max Depth
        max_depth = [int(x) for x in np.linspace(3, 8, 6)]
        
        
        subsample= [x for x in np.linspace(0.6, 0.9, 9)]
        
        params = list(itertools.product(learning_rate, subsample, 
                                        min_samples_split,
                                        max_depth))
        
        param_df = pd.DataFrame(columns=['iteration', 'learning_rate',
                                         'subsample', 'min_samples_split',
                                         'max_depth', 'score'],
                                index=range(len(params)))
        
        r = list(range(len(params)))
        random.shuffle(r)
        for random_i in r:
            if i > max_iter:
                break

            params_i = params[random_i]
            gbm = GradientBoostingClassifier(
                learning_rate=params_i[0],
                subsample=params_i[1],
                min_samples_split=params_i[2],
                max_depth=params_i[3],
                max_features='sqrt')
            
            clf = OneVsRestClassifier(gbm)
            clf.fit(X_tr, y_tr)
            score_i = clf.score(X_val, y_val)
            if score_i > max_score:
                max_score = score_i
                best_i = i
                
            param_df['iteration'][i] = i
            param_df['learning_rate'][i] = params_i[0]
            param_df['subsample'][i] = params_i[1]
            param_df['min_samples_split'][i] = params_i[2]
            param_df['max_depth'][i] = params_i[3]
            param_df['score'][i] = score_i
            i += 1
        
        params = {'learning_rate': params[best_i][0],
                  'subsample': params[best_i][1],
                  'min_samples_split': params[best_i][2],
                  'max_depth': params[best_i][3],
                  'max_features': 'sqrt'
            }
        return param_df
    
    def train_EN(self, X_tr, y_tr, X_val, y_val, split_no=0):
        max_iter = self.max_iter
        i = 0
        max_score = 0
        best_i = 0


        # learning rate
        alpha = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
        
        l1_ratio = [x for x in np.linspace(0, 1, 10)]
        
        
        params = list(itertools.product(alpha, l1_ratio))
        
        param_df = pd.DataFrame(columns=['iteration', 'alpha',
                                         'l1_ratio', 'score'],
                                index=range(len(params)))
        
        r = list(range(len(params)))
        random.shuffle(r)
        for random_i in r:
            if i > max_iter:
                break

            params_i = params[random_i]
            en = SGDClassifier(penalty='elasticnet',
                               alpha=params_i[0],
                               l1_ratio=params_i[1],
                               n_jobs=-1)
            clf = OneVsRestClassifier(en)
            clf.fit(X_tr, y_tr)
            score_i = clf.score(X_val, y_val)
            if score_i > max_score:
                max_score = score_i
                best_i = i
                
            param_df['iteration'][i] = i
            param_df['alpha'][i] = params_i[0]
            param_df['l1_ratio'][i] = params_i[1]
            param_df['score'][i] = score_i
            i += 1
        
        params = {'alpha': params[best_i][0],
                  'l1_ratio': params[best_i][1]
            }
        return param_df
    
    def best_models(self, train, test):
        all_params = self.model_params
        manipulate = DataManipulation(train, test)
            
        X_tr = manipulate.x_train.copy()
        X_test = manipulate.x_test.copy()

        y_tr = manipulate.y_train.copy()
        y_test = manipulate.y_test.copy()
        
        best_scores = {'RF': 0,
                       'GBM': 0,
                       'EN': 0}
        
        for key, df in all_params.items():
            best_params_i = df.sort_values('score', ascending=False).iloc[0,:]
            if key.startswith('RF'):
                if best_params_i['score'] > best_scores['RF']:
                    params_rf = best_params_i[1:-1]
            if key.startswith('GBM'):
                if best_params_i['score'] > best_scores['GBM']:
                    params_gbm = best_params_i[1:-1]
            if key.startswith('EN'):
                if best_params_i['score'] > best_scores['EN']:
                    params_en = best_params_i[1:-1]
        
        # RF
        rfc = RandomForestClassifier(
                n_estimators=params_rf[0],
                max_features=params_rf[1],
                min_samples_split=params_rf[2],
                min_samples_leaf=params_rf[3],
                bootstrap=params_rf[4],
                n_jobs=-1)
            
        clf = OneVsRestClassifier(rfc)
        clf.fit(X_tr, y_tr)
        self.clf_rf = clf
        score_rf = clf.score(X_test, y_test)
        with open('data/RF_model.pickle', 'wb') as f:
            pickle.dump(clf, f)
        
        # GBM
        gbm = GradientBoostingClassifier(
                learning_rate=params_gbm[0],
                subsample=params_gbm[1],
                min_samples_split=params_gbm[2],
                max_depth=params_gbm[3],
                max_features='sqrt')
            
        clf = OneVsRestClassifier(gbm)
        clf.fit(X_tr, y_tr)
        self.clf_gbm = clf
        score_gbm = clf.score(X_test, y_test)
        with open('data/GBM_model.pickle', 'wb') as f:
            pickle.dump(clf, f)
        
        # EN
        en = SGDClassifier(penalty='elasticnet',
                               alpha=params_en[0],
                               l1_ratio=params_en[1],
                               n_jobs=-1)
            
        clf = OneVsRestClassifier(en)
        clf.fit(X_tr, y_tr)
        self.clf_en = clf
        score_en = clf.score(X_test, y_test)
        with open('data/EN_model.pickle', 'wb') as f:
            pickle.dump(clf, f)
            
        print('RF:', score_rf)
        print('GBM:', score_gbm)
        print('EN:', score_en)
        
    def prediction(self, clf, model_name, train, test):
        train = train.copy()
        test = test.copy()
        manipulate = DataManipulation(train, test)
        X_tr = manipulate.x_train.copy()
        X_test = manipulate.x_test.copy()
        
        y_tr = manipulate.y_train.copy()
        y_test = manipulate.y_test.copy()
        
        # self.n_classes = manipulate.n_classes
        
        print(model_name)
        print('Train Score:')
        print(clf.score(X_tr, y_tr))
        roc_auc, fig_tr = self.getRoc(clf, model_name, X_tr, y_tr, 'Train')
        print(roc_auc)
        print('Test Score:')
        print(clf.score(X_test, y_test))
        roc_auc, fig_test = self.getRoc(clf, model_name, X_test, y_test,
                                        'Test')
        print(roc_auc)
        
        return fig_tr, fig_test
        
        
    def getRoc(self, clf, model_name, X_test, y_test, sample):
        n_classes = self.n_classes
        y_score = clf.decision_function(X_test)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(3):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(),
                                                  y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        
        # Finally average it and compute AUC
        mean_tpr /= n_classes
        
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        
        # Plot all ROC curves
        lw = 2
        fig = plt.figure()
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)
        
        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)
        
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                     ''.format(i, roc_auc[i]))
        
        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(sample)
        plt.legend(loc="lower right")
        plt.show()
        # if model_name == 'ElasticNet':
        #     self.en_ROC = roc_auc
        # elif model_name == 'RandomForest':
        #     self.rf_ROC = roc_auc
        # elif model_name == 'GBM':
        #     self.gbm_ROC = roc_auc
        return roc_auc, fig
    