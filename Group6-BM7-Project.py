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


import pandas as pd
import matplotlib.pyplot as plt
# Load the dataset
file_path = 'laptop_price - dataset.csv'  # Ensure this is the correct file path
laptop_data = pd.read_csv('laptop_price - dataset.csv')

# Scatter plot: RAM vs Price
plt.figure(figsize=(10,6))
plt.scatter(laptop_data['RAM (GB)'], laptop_data['Price (Euro)'], color='green', alpha=0.6)
plt.title('Scatter Plot of RAM vs Price')
plt.xlabel('RAM (GB)')
plt.ylabel('Price (Euro)')
plt.grid(True)

# Show the plot
plt.show()
