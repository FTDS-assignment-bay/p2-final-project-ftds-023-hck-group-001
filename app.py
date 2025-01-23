# Import Libraries
import pandas as pd
import numpy as np
import streamlit as st
import pickle
import plotly.express as px

# Load the dataset
data = pd.read_csv('cluster_kmeans.csv')

# Load the pre-trained models
with open('kmeans_model.pkl', 'rb') as f:
    kmeans_model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('pca.pkl', 'rb') as f:
    pca = pickle.load(f)

with open('pca_3d.pkl', 'rb') as f:
    pca_3d = pickle.load(f)

def eda_page():
    # Title
    st.title("Exploratory Data Analysis (EDA)")

    # Gender Distribution (Pie Chart)
    st.header("Gender Distribution")
    gender_counts = data['Gender'].value_counts()
    fig1 = px.pie(
        names=gender_counts.index,
        values=gender_counts.values,
        title="Gender Distribution",
        color_discrete_sequence=["skyblue", "lightcoral", "lightgreen"]
    )
    st.plotly_chart(fig1)

    # Remote Work Frequency by Department (Treemap)
    st.header("Remote Work Frequency by Department")
    remote_work_dept = data.groupby('Department')['Remote_Work_Frequency'].mean().reset_index()
    fig2 = px.treemap(
        remote_work_dept,
        path=['Department'],
        values='Remote_Work_Frequency',
        title="Remote Work Frequency by Department",
        color='Remote_Work_Frequency',
        color_continuous_scale='viridis'
    )
    st.plotly_chart(fig2)

    # Age Distribution (Histogram)
    st.header("Age Distribution")
    fig3 = px.histogram(
        data,
        x='Age',
        nbins=20,
        title="Age Distribution",
        color_discrete_sequence=['blue'],
        marginal="box"
    )
    st.plotly_chart(fig3)

    # Perform PCA with 3 components

    data1 = data.copy()
    data1 = data1.drop('cluster', axis=1)
    scaled_data = scaler.transform(data1.select_dtypes(include=['float64', 'int64']))
    transformed_data1 = pca_3d.transform(scaled_data)

    # Create DataFrame for PCA results
    pca_df = pd.DataFrame(transformed_data1, columns=['PCA1', 'PCA2', 'PCA3'])
    pca_df['Cluster'] = data['cluster']  # Assuming `data1` contains the cluster labels

    # 3D Scatter Plot using Plotly
    st.header("3D Cluster Visualization")

    fig_3d = px.scatter_3d(
        pca_df, 
        x='PCA1', 
        y='PCA2', 
        z='PCA3', 
        color='Cluster', 
        color_continuous_scale='Viridis',
        title='3D Cluster Visualization',
        labels={'Cluster': 'Cluster Group'}
    )

    # Customize the appearance
    fig_3d.update_traces(marker=dict(size=5, opacity=0.8))
    fig_3d.update_layout(
        scene=dict(
            xaxis_title='PCA Component 1',
            yaxis_title='PCA Component 2',
            zaxis_title='PCA Component 3'
        )
    )

    # Display plot in Streamlit
    st.plotly_chart(fig_3d)

def predict_page():
    # Title and description
    st.title('Employee Clustering')
    st.markdown("""
        **Welcome to the Employee Clustering App!**

        This application allows you to explore employee data and use a pre-trained clustering model to analyze employee characteristics.

        Fill out the fields below to predict the cluster for a new employee and receive tailored action plans based on the cluster.
    """)

    # Input new data for clustering
    st.header("Input New Employee Data for Clustering")

    with st.form(key='form_parameters'):
        # Define input fields for all columns
        employee_id = st.number_input("Employee ID", min_value=1, value=1)
        department_input = st.selectbox("Department", options=list(data["Department"].unique()))
        gender_input = st.selectbox("Gender", options=list(data["Gender"].unique()))
        age_input = st.number_input("Age", min_value=18, max_value=65, value=30)
        job_title = st.text_input("Job Title", value="Analyst")
        hire_date = st.date_input("Hire Date")
        years_at_company = st.number_input("Years at Company", min_value=0, value=1)
        education_level = st.selectbox("Education Level", options=list(data["Education_Level"].unique()))
        performance_score = st.number_input("Performance Score", min_value=1, max_value=5, value=3)
        monthly_salary = st.number_input("Monthly Salary", min_value=1000.0, value=5000.0)
        work_hours = st.number_input("Work Hours Per Week", min_value=20, max_value=60, value=40)
        projects_handled = st.number_input("Projects Handled", min_value=0, value=5)
        overtime_hours = st.number_input("Overtime Hours", min_value=0, value=10)
        sick_days = st.number_input("Sick Days", min_value=0, value=2)
        remote_work_frequency = st.number_input("Remote Work Frequency (%)", min_value=0, max_value=100, value=50)
        team_size = st.number_input("Team Size", min_value=1, value=5)
        training_hours = st.number_input("Training Hours", min_value=0, value=10)
        promotions = st.number_input("Promotions", min_value=0, value=0)
        employee_satisfaction_score = st.number_input("Employee Satisfaction Score", min_value=0.0, max_value=5.0, value=3.0)
        resigned = st.selectbox("Resigned", options=[True, False])

        # Submit button
        submit = st.form_submit_button('Submit')

    if submit:
        # Create a dataframe for the new input
        data_new = pd.DataFrame({
            "Employee_ID": [employee_id],
            "Department": [department_input],
            "Gender": [gender_input],
            "Age": [age_input],
            "Job_Title": [job_title],
            "Hire_Date": [hire_date],
            "Years_At_Company": [years_at_company],
            "Education_Level": [education_level],
            "Performance_Score": [performance_score],
            "Monthly_Salary": [monthly_salary],
            "Work_Hours_Per_Week": [work_hours],
            "Projects_Handled": [projects_handled],
            "Overtime_Hours": [overtime_hours],
            "Sick_Days": [sick_days],
            "Remote_Work_Frequency": [remote_work_frequency],
            "Team_Size": [team_size],
            "Training_Hours": [training_hours],
            "Promotions": [promotions],
            "Employee_Satisfaction_Score": [employee_satisfaction_score],
            "Resigned": [resigned]
        })

        # Preprocess the new data
        scaled_data = scaler.transform(data_new.select_dtypes(include=['float64', 'int64']))
        transformed_data = pca.transform(scaled_data)

        # Predict the cluster for the new data
        cluster_label = kmeans_model.predict(transformed_data)
        st.subheader('Prediction Result:')
        st.write(f'The predicted cluster for the input data is: Cluster {cluster_label[0]}')

        # Display tailored action plan based on the cluster
        st.subheader('Tailored Action Plan:')
        if cluster_label[0] == 0:
            st.markdown("""
            ### Cluster 0: High Performers with Lower Workload
            - **Current State**: High performance (~4.1), fewer work hours (~37/week), good satisfaction (~3.0).
            - **Challenges**: Potential underutilization due to lighter workload and high salaries.
            - **Suggestions**:
                1. Increase Project Assignments: Gradually assign more challenging and high-impact projects.
                2. Leadership Roles: Identify potential leaders in this group and provide mentorship opportunities.
                3. Recognition Programs: Reward them for consistent performance to maintain engagement.
            """)
        elif cluster_label[0] == 1:
            st.markdown("""
            ### Cluster 1: Productive Workhorses
            - **Current State**: Long hours (~53/week), high performance (~4.1), moderate satisfaction (~3.0).
            - **Challenges**: Risk of burnout due to heavy workload.
            - **Suggestions**:
                1. Focus on Retention: Implement wellness programs or perks (e.g., flexible hours, mental health days).
                2. Sustain Work-Life Balance: Avoid assigning more projects but encourage time-off or task-sharing.
                3. Upskilling: Invest in advanced training to maintain engagement and help them grow into higher roles.
            """)
        elif cluster_label[0] == 2:
            st.markdown("""
            ### Cluster 2: Moderate Hours, Low Performance
            - **Current State**: Average hours (~44/week), low performance (~2.0), moderate satisfaction (~3.0).
            - **Challenges**: Low output relative to time invested, underutilized potential.
            - **Suggestions**:
                1. Implement Performance Improvement Plans (PIPs): Include specific goals, timelines, and support (e.g., coaching, mentorship).
                2. Increase Accountability: Assign more deliverables and track results.
                3. Identify Underlying Issues: Analyze why they underperform (e.g., poor training, unclear roles) and address root causes.
            """)
        elif cluster_label[0] == 3:
            st.markdown("""
            ### Cluster 3: Inefficient and Overworked
            - **Current State**: Long overtime hours (~22.5), low performance (~2.0), low satisfaction (~2.99).
            - **Challenges**: High inefficiency and dissatisfaction despite significant effort.
            - **Suggestions**:
                1. Reassess Role Fit: Reevaluate if employees in this cluster are in roles suited to their skills. Offer role realignment if feasible.
                2. Final PIP Opportunity: Provide training and set clear, measurable improvement targets within a fixed period (e.g., 3–6 months).
                3. Layoffs for Persistent Underperformance: If no improvement is shown, consider layoffs to focus resources on more productive employees.
            """)

if __name__ == '__main__':
    # Sidebar for navigation
    page = st.sidebar.selectbox("Select Page", ["EDA", "Predict"])

    if page == "EDA":
        eda_page()
    elif page == "Predict":
        predict_page()
