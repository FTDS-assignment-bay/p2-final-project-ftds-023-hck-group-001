import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Load the dataset
data = pd.read_csv('cluster_kmeans.csv')

# Streamlit App Title
st.title("Interactive Employee Data Analysis")

# Extract year from Hire_Date if available
data['Hire_Year'] = pd.to_datetime(data['Hire_Date'], errors='coerce').dt.year

# Trend Line: Employee Satisfaction Over the Year
st.header("Trend of Employee Satisfaction Over the Years")
trend_data = data.groupby(['Hire_Year', 'cluster'])['Employee_Satisfaction_Score'].mean().reset_index()
trend_data['cluster'] = trend_data['cluster'].astype(str)
fig4, ax4 = plt.subplots(figsize=(10, 6))
for cluster in trend_data['cluster'].unique():
    cluster_data = trend_data[trend_data['cluster'] == cluster]
    ax4.plot(
        cluster_data['Hire_Year'],
        cluster_data['Employee_Satisfaction_Score'],
        marker='o',
        label=f'Cluster {cluster}'
    )
ax4.set_title('Trend of Employee Satisfaction Over the Years', fontsize=14)
ax4.set_xlabel('Year of Hire', fontsize=12)
ax4.set_ylabel('Avg Satisfaction Score', fontsize=12)
ax4.legend(title='Cluster')
ax4.grid(True, linestyle='--', alpha=0.6)
st.pyplot(fig4)

# Average Performance Score by Cluster
st.header("Average Performance Score by Cluster")
performance_data = data.groupby('cluster')['Performance_Score'].mean().reset_index()
fig5, ax5 = plt.subplots(figsize=(8, 6))
sns.barplot(data=performance_data, x='cluster', y='Performance_Score', palette='viridis', ax=ax5)
ax5.set_title('Average Performance Score by Cluster', fontsize=14)
ax5.set_xlabel('Cluster', fontsize=12)
ax5.set_ylabel('Avg Performance Score', fontsize=12)
st.pyplot(fig5)

# Stacked Bar Chart: Monthly Salary by Cluster and Gender
st.header("Monthly Salary by Cluster and Gender")
salary_gender_data = data.groupby(['cluster', 'Gender'])['Monthly_Salary'].sum().unstack()
fig6, ax6 = plt.subplots(figsize=(10, 6))
salary_gender_data.plot(
    kind='bar',
    stacked=True,
    ax=ax6,
    color=['pink','blue', 'lightgreen']
)
ax6.set_title('Monthly Salary by Cluster and Gender', fontsize=14)
ax6.set_xlabel('Cluster', fontsize=12)
ax6.set_ylabel('Avg Monthly Salary', fontsize=12)
ax6.legend(title='Gender')
st.pyplot(fig6)

# Separated Bar Charts for Key Metrics
st.header("Cluster-Wise Averages for Key Metrics")
metrics = ['Overtime_Hours', 'Projects_Handled', 'Remote_Work_Frequency', 'Work_Hours_Per_Week']
clustered_metrics = data.groupby('cluster')[metrics].mean().reset_index()
selected_metric = st.selectbox("Select a Metric to View", metrics)
fig7, ax7 = plt.subplots(figsize=(8, 6))
sns.barplot(data=clustered_metrics, x='cluster', y=selected_metric, palette='viridis', ax=ax7)
ax7.set_title(f'Average {selected_metric.replace("_", " ")}', fontsize=14)
ax7.set_xlabel('Cluster', fontsize=12)
ax7.set_ylabel(f'Avg {selected_metric.replace("_", " ")}', fontsize=12)
st.pyplot(fig7)
