# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 17:56:22 2020

@author: Fran
"""
import streamlit as st
import pandas as pd
import numpy as np
# import os
# import matplotlib.pyplot as plt
# import seaborn as sns

from datetime import datetime as dt
import utils

# Title and Subheader
st.title("Football match prediction EDA App")
st.subheader("Francisco Sanz causaLens challenge ")
data, image = utils.load_data()
st.image(image)

# Show Dataset
if st.checkbox("Preview DataFrame"):
    
	if st.button("Head"):
		st.write(data.head())
	if st.button("Tail"):
		st.write(data.tail())

if st.checkbox('Show Season Results Graphs:'):	
    option = st.selectbox('What Season would you like to see?',
                           ('Everything','14/15', '15/16', '16/17', '17/18'))
    chart1, chart2, chart3 = utils.getResults(data, season=option)
    st.bokeh_chart(chart1, use_container_width=True)
    st.bokeh_chart(chart2, use_container_width=True)
    st.bokeh_chart(chart3, use_container_width=True)
    
if st.checkbox('Show Results of models'):
    option = st.selectbox('What model would you like to see?',
                           ('Baseline with just Results',
                            'Baseline with results and odds',
                            'Baseline with everything',
                            'ElasticNet', 'GBM'))
    if option == 'Baseline with just Results':
        fig_tr, fig_test = utils.getModelsBaseline(data, option)
    elif option == 'Baseline with results and odds':
        fig_tr, fig_test = utils.getModelsBaseline(data, option)
    elif option == 'Baseline with everything':
        fig_tr, fig_test = utils.getModelsBaseline(data, option)
    else:
        fig_tr, fig_test = utils.getModels(data, option=option)
    # st.bokeh_chart(chart, use_container_width=True)
    st.write('Train sample')
    st.pyplot(fig_tr)
    st.write('Test sample')
    st.pyplot(fig_test)
# Navigate through days to see games
# dates = data['Date'].unique()
# option = st.date_input('Choose a date you want to see results for', dates[0])

# if option in dates:
#     filter_mask = data['Date'].dt.date == option
#     filtered_df = data[filter_mask]
# st.write(option)
# print(option)