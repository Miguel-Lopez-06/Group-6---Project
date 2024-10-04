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

# Assuming your data is in a pandas DataFrame named 'df'
# If your DataFrame has a different name, replace 'df' with the actual name
apple_laptops = df[df['Company'] == 'Apple']
apple_cpu_count = apple_laptops['CPU_Type'].value_counts()

# Plot
plt.figure(figsize=(10, 6))
apple_cpu_count.plot(kind='bar', color='skyblue')
plt.title('Bar Chart of Most Common CPU Types Used by Apple')
plt.xlabel('CPU Type')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
st.pyplot(plt)
# Clears the current figure
plt.clf()



