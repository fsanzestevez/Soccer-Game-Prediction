# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 19:03:53 2020

@author: Fran
"""
import pandas as pd
import numpy as np
import datetime as dt
import seaborn as sns
import os

from PIL import Image,ImageFilter,ImageEnhance

from math import pi

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from bokeh.io import push_notebook, show, output_notebook
from bokeh.palettes import Pastel1, Spectral9
from bokeh.layouts import row 
from bokeh.transform import cumsum, factor_cmap

from modelling import Modelling
from baseline import Baseline

def load_data():
    image = Image.open('data/English-Premier-League-1992-To-2020.jpg')
    image = np.asarray(image)
    data = pd.read_csv('data/fullDatabase.csv')
    data['Date'] = pd.to_datetime(data['Date'], format='%Y/%m/%d', errors = 'coerce')
    data['Date'] = data['Date'].dt.date
    return data, image


def makeDonutPlot(df, value_col='FTR'):
    if value_col == 'FTR':
        title = 'Full Time Results'
    elif value_col == 'HTR':
        title = 'Half Time Results'
    
    graph_df = pd.DataFrame()
    graph_df['Result'] = ['Home Win', 'Away Win', 'Draw']
    graph_df['Counts'] = list(df[value_col].value_counts().values)
    graph_df['Counts'] = 100*(graph_df['Counts']/graph_df['Counts'].sum())
    graph_df['color'] = Pastel1[3]
    graph_df['Angle'] = graph_df['Counts']/100 * 2*pi
    
    p = figure(plot_height=350, title=title, toolbar_location=None,
               tools="hover", tooltips="@Result: @Counts{0.2f} %",
               x_range=(-.5, .5))

    p.annular_wedge(x=0, y=1,  inner_radius=0.15, outer_radius=0.25,
                    direction="anticlock", 
                    start_angle=cumsum('Angle', include_zero=True),
                    end_angle=cumsum('Angle'), line_color="white", 
                    fill_color='color', legend='Result', source=graph_df)
    
    return p


def chart_HTvFT(df):
    df['HTvFT'] = df['HTR'] + '/' + df['FTR']
    
    labels = list(df['HTvFT'].value_counts().index)
    counts = list(df['HTvFT'].value_counts().values)
    
    source = ColumnDataSource(data=dict(labels=labels, counts=counts,
                                        color=Spectral9))
    
    p = figure(x_range=labels, y_range=(0,9), plot_height=250, 
               title="Fruit Counts", toolbar_location=None, tools="")
    
    p.vbar(x='labels', top='counts', width=0.9, color='color',
           legend_field="HTvFT", source=source)
    
    p.xgrid.grid_line_color = None
    p.legend.orientation = "horizontal"
    p.legend.location = "top_center"
    
    
    labels = list(df['HTvFT'].value_counts().index)
    counts = list(df['HTvFT'].value_counts().values)
    source = ColumnDataSource(data=dict(labels=labels,
                                        counts=counts))
    p = figure(x_range=labels, toolbar_location=None,
               title="Half Time vs Ful Time Results", tools="hover", 
               tooltips = 'Total: @counts')
    p.vbar(x='labels', top='counts', width=0.9, source=source,
           legend_field="labels", line_color='white',
           fill_color=factor_cmap('labels', palette=Spectral9,
                                  factors=labels))
    p.xgrid.grid_line_color = None
    
    return p

def getResults(data, season=None):
    if season == '14/15':
        filter_mask = data['Date'] <= pd.to_datetime('2015/07/01')
        filtered_df = data[filter_mask]
        
    if season == '15/16':
        filter_mask = (data['Date'] > pd.to_datetime('2015/07/01')) & \
                       (data['Date'] < pd.to_datetime('2016/07/01'))
        filtered_df = data[filter_mask]
        
    if season == '16/17':
        filter_mask = (data['Date'] > pd.to_datetime('2016/07/01')) & \
                       (data['Date'] < pd.to_datetime('2017/07/01'))
        filtered_df = data[filter_mask]
        
    if season == '17/18':
        filter_mask = (data['Date'] > pd.to_datetime('2017/07/01')) & \
                       (data['Date'] < pd.to_datetime('2018/07/01'))
        filtered_df = data[filter_mask]
    
    if season == 'Everything':
        filtered_df = data.copy()
    
    p1 = makeDonutPlot(filtered_df, 'FTR')
    p2 = makeDonutPlot(filtered_df, 'HTR')
    p3 = chart_HTvFT(filtered_df)
    
    return p1, p2, p3

def getModels(data, option):
    test_size = 0.15
    n = len(data)
    split_index = int(n*(1-test_size))
    train = data.iloc[:split_index, :]
    test = data.iloc[split_index:, :]
    base = Baseline(train, test)
    models = Modelling()
    if option == 'Baseline with just Results':
        clf = base.clf_essential
        fig_tr = base.fig_tr_essential
        fig_test = base.fig_test_essential
    if option == 'Baseline with results and odds':
        clf = base.clf_np
        fig_tr = base.fig_tr_np
        fig_test = base.fig_test_np
    elif option == 'Baseline with everything':
        clf = base.clf_all
        fig_tr = base.fig_tr_all
        fig_test = base.fig_test_all
    
    else:
        if option == 'ElasticNet':
            clf = pd.read_pickle('data/EN_model.pickle')
        elif option == 'GBM':
            clf = pd.read_pickle('data/GBM_model.pickle')
      
        fig_tr, fig_test = models.prediction(clf, option, train, test)

    
    return fig_tr, fig_test

def getModelsBaseline(data, option):
    test_size = 0.15
    n = len(data)
    split_index = int(n*(1-test_size))
    train = data.iloc[:split_index, :]
    test = data.iloc[split_index:, :]
    base = Baseline(train, test)
    if option == 'Baseline with just Results':
        # clf = base.clf_essential
        fig_tr = base.fig_tr_essential
        fig_test = base.fig_test_essential
    if option == 'Baseline with results and odds':
        # clf = base.clf_np
        fig_tr = base.fig_tr_np
        fig_test = base.fig_test_np
    elif option == 'Baseline with everything':
        # clf = base.clf_all
        fig_tr = base.fig_tr_all
        fig_test = base.fig_test_all

    
    return fig_tr, fig_test

    
    