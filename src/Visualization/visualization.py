# -*- coding: utf-8 -*-
import logging
import sys
sys.path.insert(0, "../Loading/")
from loading import read_csv_from_path, read_csv_from_path
logger = logging.getLogger('main_logger')
import streamlit as st
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt

st.title('Monitoring Dashboard')
st.header("Metrics")
df_eval = read_csv_from_path("../../Outputs/Monitoring/marketing_2_eval.csv")

col1, col2, col3, col4 = st.columns(4)

with col1:
    fig, ax = plt.subplots()
    st.subheader("f1_score")
    ax.plot(df_eval["count"], df_eval["f1_score"])
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    st.subheader("accuracy")
    ax.plot(df_eval["count"], df_eval["accuracy"])
    st.pyplot(fig)
with col3:
    fig, ax = plt.subplots()
    st.subheader("recall")
    ax.plot(df_eval["count"], df_eval["recall"])
    st.pyplot(fig)
with col4:
    fig, ax = plt.subplots()
    st.subheader("precision")
    ax.plot(df_eval["count"],df_eval["precision"])
    st.pyplot(fig)

st.header("Tests")
df_monitoring = read_csv_from_path("../../Outputs/Monitoring/marketing_2_monitored.csv")
option_batch = st.selectbox('Batch',[1,2,3,4])
df_monitoring_batch=df_monitoring[df_monitoring["batch"]==option_batch]
df_monitoring_batch_pivot=df_monitoring_batch.pivot(index='col', columns='test', values='val')
cmap = sns.diverging_palette(133, 10, as_cmap=True)
fig, ax = plt.subplots()
sns.heatmap(df_monitoring_batch_pivot, ax=ax,cbar=False,cmap=cmap,linewidths=.5)
st.write(fig)
st.header("Graphs")
df = read_csv_from_path("../../Outputs/Preprocessed/marketing_preprocessed.csv")
cols = df.columns
option = st.selectbox('feature',cols)
train = df[option]

df2 = read_csv_from_path("../../Outputs/Preprocessed/marketing_2_preprocessed.csv")[option]
data = np.array_split(df2,4)+[train]

fig = ff.create_distplot(data,["batch_1","batch_2","batch_3","batch_4","train"])
st.plotly_chart(fig, use_container_width=True)
