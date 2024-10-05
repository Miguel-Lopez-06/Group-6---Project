import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import altair as alt
from wordcloud import WordCloud
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
import plotly.graph_objects as go
from io import StringIO

st.title('Data Visualization Using Streamlit')
st.markdown('`Group 6 BM7`')
st.header('------------------------------------------------------------')


df = pd.read_csv("laptop_price-dataset.csv")

df

st.header('------------------------------------------------------------')
df.info

st.header('------------------------------------------------------------')
#Graph 1
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

st.write('This Bar Chart shows the types of CPU that Apple used in their laptops.')
st.write('**Observation**: Intel Core i5: Because of the balance between the performance and efficiency of this CPU, it is the most used CPU of Apple in their MacBooks. Intel Core i7: This CPU appears more on the higher-end models of Macbook like the Macbook Pro.')
st.header('------------------------------------------------------------')

#Graph2
st.header('Violin Plot of Weight Distribution of Laptops')

import seaborn as sns
import matplotlib.pyplot as plt

# 3. Violin Plot for Weight Distribution of Laptops
plt.figure(figsize=(10, 6))  # Set the size of the plot

# Assuming your DataFrame is named 'df'
sns.violinplot(data=df, x='Weight (kg)', color='pink')  # Create a violin plot for laptop weights

plt.title('Violin Plot of Weight Distribution of Laptops')  # Set the title of the plot
plt.xlabel('Weight (kg)')  # Label the x-axis
plt.tight_layout()  # Adjust layout for better visualization
st.pyplot(plt)
# Clears the current figure
plt.clf()

st.write('The weight distribution of laptops is displayed in this violin plot, emphasizing the areas where the bulk of computers fall within a particular weight range.')
st.write('**Observation:** The majority of computers weigh between 1.0 and 2.5 kg, with bulkier models weighing between 1.8 kg and 1.3–1.5 kg at their highest points. Ultra-heavy laptops are rare, as only a small percentage of computers weigh more than 2.5 kg.')
st.header('------------------------------------------------------------')

#Graph3
st.header('Bubble Chart: RAM vs CPU Frequency vs Price')

import matplotlib.pyplot as plt

# Bubble Chart: RAM vs CPU Frequency vs Price
plt.figure(figsize=(10, 6))
plt.scatter(df['CPU_Frequency (GHz)'], df['RAM (GB)'],
            s=df['Price (Euro)']/10, alpha=0.6,
            c=df['Price (Euro)'], cmap='coolwarm', marker='o')

plt.title('Bubble Chart: RAM vs CPU Frequency vs Price', fontsize=14)
plt.xlabel('CPU Frequency (GHz)', fontsize=12)
plt.ylabel('RAM (GB)', fontsize=12)
plt.colorbar(label='Price (Euro)')
plt.grid(True)
st.pyplot(plt)
# Clears the current figure
plt.clf()

st.write('This bubble chart reveals a clear correlation between price and performance (RAM and CPU frequency) can be seen in this bubble chart. Since premium components are more expensive, high-end laptops are priced much higher, while budget versions with lower performance are less expensive.')
st.write('**Observation:** Generally speaking, laptops with CPU frequencies higher than 3 GHz and more RAM than 16 GB cost more. These are probably meant for customers who need a lot of processing power, such developers, gamers, or people who work on graphics-intensive projects.')
st.header('------------------------------------------------------------')

#Graph4
st.header('Area Chart: Average Laptop Price by Company')

# Area Chart: Distribution of Prices across Companies
df_grouped = df.groupby('Company')['Price (Euro)'].mean().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
df_grouped.plot(kind='area', color='skyblue', alpha=0.6)
plt.title('Area Chart: Average Laptop Price by Company', fontsize=14)
plt.xlabel('Company', fontsize=12)
plt.ylabel('Average Price (Euro)', fontsize=12)
plt.grid(True)
st.pyplot(plt)
# Clears the current figure
plt.clf()

st.write('The area chart illustrates the varied price tactics used by different laptop manufacturers. As a luxury tech brand, Apple places a premium price point on its products, but other firms target a wider market with a variety of products that range from high-end laptops to low-cost models.')
st.write('**Observation:** When it comes to typical laptop pricing, Apple leads the competition by a wide margin. This is a reflection of Apple focus on the premium end of the market, where devices like the MacBook Pro and MacBook Air are renowned for their luxury construction, cutting-edge features, and devoted customer bases.')
st.header('------------------------------------------------------------')
