# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 21:20:39 2020

@author: Fran
"""


import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype


class DataManipulation():
    def __init__(self, data, target='FTR', train=True):

        if train:
            data, cols = self.removeGameStatistics(data)
            self.delCols = cols
            data, cols = self.removeNames(data)
            self.delCols += cols
            data, cols = self.removeEmpty(data)
            self.delCols += cols
            
            data, self.imputeDict = self.imputeMissing(data, target)
            
            self.y_train = data[target].copy()
            self.x_train = data.drop(labels=target, axis=1, inplace=False)
        else:
            data.drop(labels=self.delCols, axis=1, inplace=True)
            data, _ = self.imputeMissing(data, target, self.imputeDict)
            self.y_test = data[target].copy()
            self.x_test = data.drop(labels=target, axis=1, inplace=False)
        
    
    def removeGameStatistics(data, ht=True):
        '''
        We need to drop features of the game that we can't get before the game
        has taken place, like shots, passes, possesion and so.
        There is a flag variable to indicate whether we want to predict the 
        result at half time or before the game starts.

        Parameters
        ----------
        data : TYPE pandas.DataFrame
            DESCRIPTION. collected data 
        ht : TYPE, optional
            DESCRIPTION. The default is True. flag to indicate half time

        Returns
        -------
        data : TYPE pandas.DataFrame
            DESCRIPTION. Data without in-game statistics
        columns : TYPE list
            DESCRIPTION. columns that have been deleted, therefore will need
            to be deleted in the test set

        '''
        
        columns = ['FTHG', 'FTAG','HTHG', 'HTAG', 'HTR', 'HS', 'AS', 'HST',
                   'AST', 'HHW', 'AHW', 'HC', 'AC', 'HF', 'AF', 'HFKC', 'AFKC',
                   'HO', 'AO', 'HY', 'AY', 'HR', 'AR', 'HBP', 'ABP']
        if ht:
            halfTime = ['HTHG', 'HTAG', 'HTR']
            [columns.remove(x) for x in halfTime] 
        
        data.drop(labels=columns, axis=1, inplace=True)

        return data, columns
    
    
    def removeNames(data):
        '''
        The players' names don't add value, since we have their overall ratings

        Parameters
        ----------
        data : TYPE pandas.DataFrame
            DESCRIPTION. collected data

        Returns
        -------
        data : TYPE pandas.DataFrame
            DESCRIPTION. data without players' names
        cols : TYPE list
            DESCRIPTION. columns that have been deleted, therefore will need
            to be deleted in the test set

        '''
        cols = []
        for x in data.columns:
            if str(x).startswith('Player'):
                cols.append(x)
        
        data.drop(labels=cols, axis=1, inplace=True)

        return data, cols


    def removeEmpty(data, target, threshold = 0.3):
        '''
        Delete rows/columns that are empty more than threshold % indicated

        Parameters
        ----------
        data : TYPE pandas.DataFrame
            DESCRIPTION. collected data
        target: TYPE string
            DESCRIPTION. name of the target variable. We'll delete any row
            with a missing value in the target.
        threshold : TYPE, optional
            DESCRIPTION. The default is 0.5.

        Returns
        -------
        data : TYPE pandas.DataFrame
            DESCRIPTION. data with no empty rows/columns
        delCols : TYPE list
            DESCRIPTION. columns that have been deleted, therefore will need
            to be deleted in the test set

        '''
        delCols = []
        no_rows = data.shape[0]
        no_cols = data.shape[1]
        thresh_col = int(no_rows * threshold) # drop a col if threshold % is empty
        thresh_row = int(no_cols * threshold) # drop a row if threshold % is empty
         
        rows = data.isnull().sum(axis=1)
        [data.drop(labels=i, axis=0, inplace=True) for i, v in \
         rows.iteritems() if v >= thresh_row]
        
        cols = data.isnull().sum(axis=0)
        for i, v in cols.iteritems():
            if v >= thresh_col:
                delCols.append(i)
        data.drop(labels=delCols, axis=1, inplace=True)
        
        return data, delCols

    
    def imputeMissing(data, target, imputeDict=None):
        '''
        Impute missing with mode, mean depending on data type

        Parameters
        ----------
        data : TYPE pandas.DataFrame
            DESCRIPTION. collected data
        target: TYPE string
            DESCRIPTION. name of the target variable. We can't impute values 
            into the target variable.
        imputeDict : TYPE, optional dictionary 
            DESCRIPTION. The default is None. dictionary that indicates the 
            imputing value of each column

        Returns
        -------
        data : TYPE pandas.DataFrame
            DESCRIPTION. data with no missing values
        imputeDict : TYPE, optional dictionary 
            DESCRIPTION. The default is None. dictionary that indicates the 
            imputing value of each column

        '''
        if not imputeDict:
            imputeDict = dict()
        
            for col in data.columns:
                if col == target:
                    continue
                if is_string_dtype(data[col]):
                    imputeDict[col] = data[col].mode()[0]
                elif is_numeric_dtype(data[col]):
                    imputeDict[col] = data[col].mean()
        
        data.fillna(imputeDict, inplace=True)
        
        return data, imputeDict
        