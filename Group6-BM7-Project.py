import matplotlib
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import altair as alt
from wordcloud import WordCloud
from mpl_toolkits.mplot3d import Axes3D
import squarify
import networkx as nx
import plotly.graph_objects as go
from io import StringIO

st.title('Data Visualization Using Streamlit')
st.markdown('by Group 6 BM7')
st.header('------------------------------------------------------------')


df = pd.read_csv("laptop_price - dataset.csv")

df

st.header('------------------------------------------------------------')
df.info

st.header('------------------------------------------------------------')
st.header('Bar Chart of Most Common CPU Types Used by Apple')


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd

# Assuming df is your DataFrame and is already loaded
# For example: df = pd.read_csv('your_dataset.csv')

# 3D Surface Plot: Screen Size, Weight, and Price
fig = plt.figure(figsize=(10, 11))
ax = fig.add_subplot(111, projection='3d')

# Sample data from your dataframe (df) assuming df is defined
X = df['Inches']
Y = df['Weight (kg)']
Z = df['Price (Euro)']
ax.plot_trisurf(X, Y, Z, cmap='viridis', edgecolor='none')

# Set titles and labels
ax.set_title('3D Surface Plot: Screen Size, Weight, and Price', fontsize=14)
ax.set_xlabel('Screen Size (Inches)', fontsize=10)
ax.set_ylabel('Weight (kg)', fontsize=10)
ax.set_zlabel('Price (Euro)', fontsize=10)

plt.show()

