# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 21:20:39 2020

@author: Fran
"""


import pandas as pd
import datetime as dt
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize


class DataManipulation():
    def __init__(self, train = pd.DataFrame(), test = pd.DataFrame(),
                 target='FTR'):

        if not train.empty:
            train = self.create_features(train)
            train['Date'] = train['Date'].map(dt.datetime.toordinal)
            train, cols = self.removeGameStatistics(train)
            self.delCols = cols
            
            train, cols = self.removeNames(train)
            self.delCols += cols
            
            train, cols = self.removeEmpty(train, target)
            self.delCols += cols
            
            train, cols = self.removeConstants(train)
            self.delCols += cols
            
            train, self.imputeDict = self.imputeMissing(train, target)
            
            # self.le = LabelEncoder()
            y_train = train[target].copy()
            y_train = label_binarize(y_train, classes=['H','A','D'])
            self.n_classes = 3
            self.y_train = y_train
            # self.y_train = self.le.fit_transform(y_train)
            train.drop(labels=target, axis=1, inplace=True)
            train = self.scaleDF(train, isTrain=True)
            self.x_train = pd.get_dummies(train)
            
            
        if not test.empty:
            test = self.create_features(test)
            test = test[test[target].notna()]
            test['Date'] = test['Date'].map(dt.datetime.toordinal)
            test = self.dropCsTest(test)
            test, _ = self.imputeMissing(test, target, self.imputeDict)
            y_test = test[target].copy()
            # self.y_test = self.le.transform(y_test)
            y_test = label_binarize(y_test, classes=['H','A','D'])
            self.n_classes = 3
            self.y_test = y_test
            test.drop(labels=target, axis=1, inplace=True)
            test = pd.get_dummies(test)
            test = self.columnConsistency(test)
            self.x_test = self.scaleDF(test, isTrain=False)
            
        
    
    def removeGameStatistics(self, df, pred_time='HalfTime'):
        '''
        We need to drop features of the game that we can't get before the game
        has taken place, like shots, passes, possesion and so.
        There is a flag variable to indicate whether we want to predict the 
        result at half time or before the game starts.

        Parameters
        ----------
        df : TYPE pandas.DataFrame
            DESCRIPTION. collected data 
        pred_time : TYPE, optional. String
            DESCRIPTION. The default is 'HalfTime'. variable to indicate
            whether we want to predict the result at half time or before the  
            game starts

        Returns
        -------
        df : TYPE pandas.DataFrame
            DESCRIPTION. Data without in-game statistics
        columns : TYPE list
            DESCRIPTION. columns that have been deleted, therefore will need
            to be deleted in the test set

        '''
        columns = ['FTHG', 'FTAG','HTHG', 'HTAG', 'HTR', 'HS', 'AS', 'HST',
                   'AST', 'HHW', 'AHW', 'HC', 'AC', 'HF', 'AF', 'HFKC', 'AFKC',
                   'HO', 'AO', 'HY', 'AY', 'HR', 'AR', 'HBP', 'ABP']
        if pred_time == 'HalfTime':
            halfTime = ['HTHG', 'HTAG', 'HTR']
            [columns.remove(x) for x in halfTime] 
        
        for col in columns:
            try:
                df.drop(labels=col, axis=1, inplace=True)
            except KeyError:
                pass
        
        return df, columns
    
    
    def removeNames(self, df):
        '''
        The players' names don't add value, since we have their overall ratings
        Home and Away team variables are also repeated.

        Parameters
        ----------
        df : TYPE pandas.DataFrame
            DESCRIPTION. collected data

        Returns
        -------
        df : TYPE pandas.DataFrame
            DESCRIPTION. data without players' names
        cols : TYPE list
            DESCRIPTION. columns that have been deleted, therefore will need
            to be deleted in the test set

        '''
        cols = []
        for x in df.columns:
            if (str(x).startswith('Player')) or (str(x).startswith('Team')):
                cols.append(x)
        
        df.drop(labels=cols, axis=1, inplace=True)

        return df, cols


    def removeEmpty(self, df, target, threshold = 0.3):
        '''
        Delete rows/columns that are empty more than threshold % indicated

        Parameters
        ----------
        df : TYPE pandas.DataFrame
            DESCRIPTION. collected data
        target: TYPE string
            DESCRIPTION. name of the target variable. We'll delete any row
            with a missing value in the target.
        threshold : TYPE, optional
            DESCRIPTION. The default is 0.5.

        Returns
        -------
        df : TYPE pandas.DataFrame
            DESCRIPTION. data with no empty rows/columns
        delCols : TYPE list
            DESCRIPTION. columns that have been deleted, therefore will need
            to be deleted in the test set

        '''
        df = df[df[target].notna()]
        delCols = []
        no_rows = df.shape[0]
        no_cols = df.shape[1]
        thresh_col = int(no_rows * threshold) # drop a col if threshold % is empty
        thresh_row = int(no_cols * threshold) # drop a row if threshold % is empty
         
        rows = df.isnull().sum(axis=1)
        [df.drop(labels=i, axis=0, inplace=True) for i, v in \
         rows.iteritems() if v >= thresh_row]
        
        cols = df.isnull().sum(axis=0)
        for i, v in cols.iteritems():
            if v >= thresh_col:
                delCols.append(i)
        df.drop(labels=delCols, axis=1, inplace=True)
        
        return df, delCols

    def removeConstants(self, df):
        '''
        Remove constant columns

        Parameters
        ----------
        df : TYPE pandas.DataFrame
            DESCRIPTION. data that may have constant columns

        Returns
        -------
        df : TYPE pandas.DataFrame
            DESCRIPTION. data without constant columns
        delCols : TYPE list
            DESCRIPTION. columns that have been dropped in this method

        '''
        oldCols = df.columns.tolist()
        df = df.loc[:, (df != df.iloc[0]).any()]
        newCols = df.columns.tolist()
        delCols = [x for x in oldCols if x not in newCols]
        
        return df, delCols
    
    
    def imputeMissing(self, df, target, imputeDict=None):
        '''
        Impute missing with mode, mean depending on data type

        Parameters
        ----------
        df : TYPE pandas.DataFrame
            DESCRIPTION. collected data
        target: TYPE string
            DESCRIPTION. name of the target variable. We can't impute values 
            into the target variable.
        imputeDict : TYPE, optional dictionary 
            DESCRIPTION. The default is None. dictionary that indicates the 
            imputing value of each column

        Returns
        -------
        df : TYPE pandas.DataFrame
            DESCRIPTION. data with no missing values
        imputeDict : TYPE, optional dictionary 
            DESCRIPTION. The default is None. dictionary that indicates the 
            imputing value of each column

        '''
        if not imputeDict:
            imputeDict = dict()
        
            for col in df.columns:
                if col == target:
                    continue
                if is_string_dtype(df[col]):
                    imputeDict[col] = df[col].mode()[0]
                elif is_numeric_dtype(df[col]):
                    imputeDict[col] = df[col].mean()
        
        df.fillna(imputeDict, inplace=True)
        
        return df, imputeDict
    
    def create_features(self, df):
        try:
            df['Date'].dt
        except AttributeError:
            try:
                df['Date'] = df['Date'].map(dt.datetime.fromordinal)
                df['Date'] = pd.to_datetime(df['Date'], errors = 'coerce')
            except TypeError:
                df['Date'] = pd.to_datetime(df['Date'], errors = 'coerce')
        
        
        df['hour'] = df['Date'].dt.hour
        df['dayofweek'] = df['Date'].dt.dayofweek
        df['quarter'] = df['Date'].dt.quarter
        df['month'] = df['Date'].dt.month
        df['year'] = df['Date'].dt.year
        df['dayofyear'] = df['Date'].dt.dayofyear
        df['dayofmonth'] = df['Date'].dt.day
        df['weekofyear'] = df['Date'].dt.weekofyear
        
        return df

    def scaleDF(self, df, isTrain=True):
        if isTrain:
            self.scaledCols = [col for col in df.columns \
                               if is_numeric_dtype(df[col])]
            self.scaler = StandardScaler()
            self.scaler.fit(df[self.scaledCols])
            
        df[self.scaledCols] = self.scaler.transform(df[self.scaledCols])
        return df
                    
    def dropCsTest(self, df):
        '''
        Drop columns in Test. We drop the same columns than on the train sample

        Parameters
        ----------
        df : TYPE pandas.DataFrame
            DESCRIPTION.

        Returns
        -------
        df : TYPE pandas.DataFrame
            DESCRIPTION.

        '''
        for col in self.delCols:
            try:
                df.drop(labels=col, axis=1, inplace=True)
            except KeyError:
                pass
        return df
    
    def columnConsistency(self, test):
        trainCols = self.x_train.columns.tolist()
        testCols = test.columns.tolist()
        
        keepCols = [x for x in testCols if x in trainCols]
        test = test[keepCols]
        
        # Columns in train that are not in test
        createCols = list(set(trainCols).difference(testCols))
        for col in createCols:
            test[col] = 0
        
        return test