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


df = pd.read_csv("laptop_price-dataset.csv")

df

st.header('------------------------------------------------------------')
df.info

st.header('------------------------------------------------------------')
st.header('Bar Chart of Most Common CPU Types Used by Apple')

import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'laptop_price-dataset.csv'  # Replace with your file path in Google Colab
laptop_data = pd.read_csv("laptop_price-dataset.csv")

# Create a bar chart for the distribution of laptop types (TypeName)
plt.figure(figsize=(10,6))
laptop_data['TypeName'].value_counts().plot(kind='bar', color='orange', edgecolor='black')
plt.title('Bar Chart of Distribution of Laptop Types')
plt.xlabel('Laptop Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.grid(True)
st.pyplot(plt)
# Clears the current figure
plt.clf()



