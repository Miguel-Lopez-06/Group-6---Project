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


from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

# Uncomment this if you're in Jupyter Notebook or using VS Code Interactive window
# %matplotlib notebook  # For interactive plot
# %matplotlib inline    # For static images (Jupyter Notebook)

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

# Save the plot as a PNG file
plt.savefig('3d_surface_plot.png')  # Save the plot to a file
print("Plot saved as '3d_surface_plot.png'")

# Use plt.show() if the environment supports interactive plotting
try:
    plt.show()
except Exception as e:
    print(f"Error showing plot interactively: {e}")

