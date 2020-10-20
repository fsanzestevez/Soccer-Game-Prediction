# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 10:56:22 2020

@author: Fran
"""

from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier as LR
from dataManipulation import DataManipulation
from modelling import Modelling

class Baseline():
    def __init__(self, train, test):
        self.train = train.copy()
        self.test = test.copy()
        train_esntl, test_esntl = self.keepEssentials()
        self.make_clf(train_esntl, test_esntl, 'Essentials')
        train_noPlyrs, test_noPlyrs = self.noPlayers()
        self.make_clf(train_noPlyrs, test_noPlyrs, 'No Players')
        self.make_clf(self.train, self.test, 'Everything')
        
    def keepEssentials(self):
        esntl_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTR']
        train_esntl = self.train[esntl_cols]
        test_esntl = self.test[esntl_cols]
        return train_esntl, test_esntl
    
    def noPlayers(self):
        columns = list(self.train.columns)
        columns = [x for x in columns if ('Player' not in x) & ('P_' not in x)]
        train_noPlyrs = self.train[columns]
        test_noPlyrs = self.test[columns]
        return train_noPlyrs, test_noPlyrs
        
    def make_clf(self, train, test, step):
        print(step)
        manipulate = DataManipulation(train, test)
        X_tr = manipulate.x_train.copy()
        X_test = manipulate.x_test.copy()

        y_tr = manipulate.y_train.copy()
        y_test = manipulate.y_test.copy()
        
        lr = LR()
        clf = OneVsRestClassifier(lr)
        clf.fit(X_tr, y_tr)
        
        models = Modelling()
        roc_auc_tr, fig_tr = models.getRoc(clf, 'BaseLine', X_tr, y_tr)
        self.roc_auc_tr = roc_auc_tr
        score_tr = clf.score(X_tr, y_tr) 
        score_test = clf.score(X_test, y_test) 
        roc_auc_test, fig_test = models.getRoc(clf, 'BaseLine', X_test, y_test)
        self.roc_auc_test = roc_auc_test
        if step == 'Essentials':
            self.clf_essential = clf
            self.fig_tr_essential = fig_tr
            self.fig_test_essential = fig_test
        elif step == 'No Players':
            self.clf_np = clf
            self.fig_tr_np = fig_tr
            self.fig_test_np = fig_test
        elif step == 'Everything':
            self.clf_all = clf
            self.fig_tr_all = fig_tr
            self.fig_test_all = fig_test
        print('Train set:', score_tr, '\nTest set:', score_test,'\n')
        print('Train')
        print('micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc_tr["micro"]))
        print('macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc_tr["macro"]))
        print('Test')
        print('micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc_test["micro"]))
        print('macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc_test["macro"]))
        
        print('\n====================================\n')