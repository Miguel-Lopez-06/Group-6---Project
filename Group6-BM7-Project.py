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

#Graph5
st.header('3D Surface Plot: Screen Size, Weight, and Price')

from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# 3D Surface Plot: Screen Size, Weight, and Price
fig = plt.figure(figsize=(10, 11))
ax = fig.add_subplot(111, projection='3d')

X = df['Inches']
Y = df['Weight (kg)']
Z = df['Price (Euro)']
ax.plot_trisurf(X, Y, Z, cmap='viridis', edgecolor='none')

ax.set_title('3D Surface Plot: Screen Size, Weight, and Price', fontsize=14)
ax.set_xlabel('Screen Size (Inches)', fontsize=10)
ax.set_ylabel('Weight (kg)', fontsize=10)
ax.set_zlabel('Price (Euro)', fontsize=10)
st.pyplot(plt)
# Clears the current figure
plt.clf()

st.write('The 3D surface plot shows that weight and screen size are reliable indicators of cost. Generally speaking, larger, heavier laptops have more powerful hardware, which drives up the price. On the other hand, portability is a common goal of lighter, smaller laptops, the cost of which varies based on the brand and internal parts.')
st.write('**Observation:** Higher-end laptops typically have larger screens (15+ inches) and weigh more (2+ kg). This implies that more expensive parts, such sharper screens, potent GPUs, and premium materials, are frequently found in larger laptops.')
st.header('------------------------------------------------------------')

#Graph6
st.header('Bar Chart of Distribution of Laptop Types')

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

st.write('This is a Bar graph of the distribution of laptop types.')
st.write('**Observation:** In this graph, there are different laptop types namingly Notebook, Gaming, Ultrabook, Netbook, and other more. Notebook has a visible lead among all the laptop types in terms of distribution, as the distribution is around 700. Next is Gaming and Ultrabook laptops which have around 150-250 distribution. Workstation and Netbook had the two lowest distributions of laptops with less than 100. Notebook is one of the most famous types of laptops. Notebooks are lightweight and smaller which makes it portable and good for basic tasks like web browsing. Gaming laptops are designed to handle playing video games. It is also good for video editing because of the higher RAM and better GPU than other laptops.')
st.header('------------------------------------------------------------')

#Graph7
st.header('Scatter Plot of RAM vs Price')

import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'laptop_price-dataset.csv'  # Replace with your file path in Google Colab
laptop_data = pd.read_csv("laptop_price-dataset.csv")

# Scatter plot: RAM vs Price
plt.figure(figsize=(10,6))
plt.scatter(laptop_data['RAM (GB)'], laptop_data['Price (Euro)'], color='green', alpha=0.6)
plt.title('Scatter Plot of RAM vs Price')
plt.xlabel('RAM (GB)')
plt.ylabel('Price (Euro)')
plt.grid(True)
st.pyplot(plt)
# Clears the current figure
plt.clf()

st.write('This is a scatter plot of RAM vs Price')
st.write('**Observation:** The graph shows a positive correlation between the Price and RAM as the RAM increases the price of the laptop also increases. 32 GB RAM laptops are the most expensive out of the other RAM choices. One outlier is the 64 GB RAM laptop costs around 4,000 euros which is slightly cheaper than other 32 GB RAM laptops. The price of the laptop also varies based on the different components installed or features.')
st.header('------------------------------------------------------------')

#Graph8
st.header('Line Chart of Average Laptop Price by Operating System')

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import streamlit as st

# Convert Price (Euro) to numeric if necessary
df['Price (Euro)'] = pd.to_numeric(df['Price (Euro)'], errors='coerce')

# Drop rows with missing prices or operating system information
df = df.dropna(subset=['Price (Euro)', 'OpSys'])

# Group the data by operating system and calculate the average price
avg_price_per_os = df.groupby('OpSys')['Price (Euro)'].mean().sort_values(ascending=False)

# Plot the Area Chart for Average Laptop Price by Operating System
plt.figure(figsize=(10, 6))
avg_price_per_os.plot(kind='area', color='skyblue', alpha=0.6)
plt.title('Area Chart: Average Laptop Price by Operating System', fontsize=14)
plt.xlabel('Operating System', fontsize=12)
plt.ylabel('Average Price (Euro)', fontsize=12)
plt.grid(True)

# Display the plot using Streamlit
st.pyplot(plt)

# Clears the current figure
plt.clf()


st.write('This line chart compares the average price of laptops running different operating systems.')
st.write('**Observation:** Laptops that run on Android OS, Chrome OS, Linux, or have no operating system at all are significantly cheaper, while Windows laptops and MacBooks are more expensive. It is also notable that laptops that run on Windows 7 are more expensive than ones that run on Windows 10 and 10 S.')
st.header('------------------------------------------------------------')



#Graph10
st.header('Area Chart: Average Laptop Price by Company')

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the CSV file
file_path = '/mnt/data/laptop_price-dataset.csv'
laptop_data = pd.read_csv('laptop_price-dataset.csv')

# Convert Price (Euro) to numeric if necessary (in case it's read as string)
df['Price (Euro)'] = pd.to_numeric(df['Price (Euro)'], errors='coerce') # Use df instead of laptop_data

# Drop rows with missing prices (if any)
df = df.dropna(subset=['Price (Euro)']) # Use df instead of laptop_data

# Histogram: Distribution of laptop prices
def generate_histogram(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Price (Euro)'], bins=30, kde=True, color='skyblue')
    plt.title('Histogram of Distribution of Laptop Prices')
    plt.xlabel('Price (Euro)')
    plt.ylabel('Frequency')
    plt.grid(True)
st.pyplot(plt)
# Clears the current figure
plt.clf()
generate_histogram(df)

st.write('The area chart illustrates the varied price tactics used by different laptop manufacturers. As a luxury tech brand, Apple places a premium price point on its products, but other firms target a wider market with a variety of products that range from high-end laptops to low-cost models.')
st.write('**Observation:** When it comes to typical laptop pricing, Apple leads the competition by a wide margin. This is a reflection of Apple focus on the premium end of the market, where devices like the MacBook Pro and MacBook Air are renowned for their luxury construction, cutting-edge features, and devoted customer bases.')
st.header('------------------------------------------------------------')