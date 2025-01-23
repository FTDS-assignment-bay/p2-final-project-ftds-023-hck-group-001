import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import squarify  # For treemap
import streamlit as st

# Load the dataset
data = pd.read_csv('cluster_kmeans.csv')

# Streamlit App Title
st.title("Interactive Employee Data Analysis")

# Gender Distribution (Pie Chart)
st.header("Gender Distribution")
gender_counts = data['Gender'].value_counts()
fig1, ax1 = plt.subplots(figsize=(6, 6))
ax1.pie(
    gender_counts,
    labels=gender_counts.index,
    autopct='%1.1f%%',
    startangle=140,
    colors=['skyblue', 'lightcoral', 'lightgreen']
)
ax1.set_title('Gender Distribution', fontsize=14)
st.pyplot(fig1)

# Remote Work Frequency by Department (Treemap)
st.header("Remote Work Frequency by Department")
remote_work_dept = data.groupby('Department')['Remote_Work_Frequency'].mean()
treemap_data = pd.DataFrame({'Department': remote_work_dept.index, 'Frequency': remote_work_dept.values})
fig2, ax2 = plt.subplots(figsize=(12, 6))
squarify.plot(
    sizes=treemap_data['Frequency'],
    label=treemap_data['Department'],
    alpha=0.8,
    color=sns.color_palette('viridis', len(treemap_data))
)
ax2.axis('off')
ax2.set_title('Remote Work Frequency by Department', fontsize=14)
st.pyplot(fig2)

# Age Distribution (Histogram)
st.header("Age Distribution")
fig3, ax3 = plt.subplots(figsize=(8, 6))
sns.histplot(data['Age'], bins=20, kde=True, color='blue', ax=ax3)
ax3.set_title('Age Distribution', fontsize=14)
ax3.set_xlabel('Age', fontsize=12)
ax3.set_ylabel('Frequency', fontsize=12)
st.pyplot(fig3)

