# -*- coding: utf-8 -*-
import logging
import sys
sys.path.insert(0,"../Loading/")
import loading
logger = logging.getLogger('main_logger')
import streamlit as st
import pandas as pd
import numpy as np
import plotly.figure_factory as ff

st.title('Monitoring Dashboard')
df= loading.read_csv_from_path("../../Outputs/Preprocessed/marketing_preprocessed.csv")
cols=df.columns
option = st.selectbox('feature',cols)
train=df[option]
df2= loading.read_csv_from_path("../../Outputs/Preprocessed/marketing_2_preprocessed.csv")[option]
data=np.array_split(df2,4)+[train]
fig = ff.create_distplot(data,["batch_1","batch_2","batch_3","batch_4","train"])
st.plotly_chart(fig, use_container_width=True)





