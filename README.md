# 📊 Project Title: Employee Clustering
## 📝 Overview
This is the final group project where we create a Machine Learning Clustering algorithm that can group employees in our data based on their performance and other factors such as hours worked, salary, overtime, etc. The cluster numbers that we found to be appropriate by considering business knowledge and the machine's statistical preference are 4. In the later section, we will give recommendation on how managers and HR teams can do to these groups of employees.

## Deployment Link (Hugging Face):
https://huggingface.co/spaces/Aymenjb/Finpro_Employee_Clustering 

## 📂 Dataset
Source: https://www.kaggle.com/datasets/mexwell/employee-performance-and-productivity-data
Size: 100000 rows × 20 columns

Description: This is an employee performance dataset that we gathered from Kaggle

## 🔧 Technologies Used
Programming Language: Python

Libraries: Pandas, Seaborn, Matplotlib, NumPy, Airflow, PythonGX, elasticsearch, Scipy, ScikitLearn

Tools: Jupyter Notebook, SQL, Docker, ElasticNet, Kibana, Airflow

## 🚀 Workflow
### 1️ Data Collecting
This is the part where we download the data from Kaggle and move it to the SQL Database
### 2 Data Extraction, Transformation, and Load inside the DAG file
This DAG file is responsible to create an ETL process of the data, which are:
1. Extraction
- This is where the raw data is extracted from my SQL database using query
2. Transformation
- This is the part where we clean the data. However, there is not much to be cleaned apart from dropping duplicates and filtering the employees who have not resigned yet
- The final product is in the file named `data_selected.csv`
3. Loading
- The clean version of the data then is created and ready to be used both in the repository and ElasticNet database
### 3 EDA 1
This exploratory data analysis is done to gather insight for Feature Engineering part. Here are the insight we have gathered:
1. We found no missing values
2. No outliers were found
3. No correlation were found too
Hence, on paper this data seems to be perfect
### 4 Feature Engineering
We decide that for this model, it is better to use KMeans Clustering because the data just seems to not have any anomialies from outlier or missing values. In feature engineering we did dimesonality reduction where we intend to keep 12 features from our dataset for the model.
### 5 Modelling
During the model, we also evaluate the appropriate number of clusters via Silhouette Plot. After plotting it, we decide that 4 is the best number. After the modelling, we export the clustered data into a dataframe and export the model into pkl file. The final data product is `cluster_kmeans.csv` and the final model is `kmeans_model.pkl`.
### 6 EDA 2 & Suggestions
This is the final EDA where we gather insight regarding each clusters that our model has created:
1. Cluster 0: This group of employees show high efficiency if we take a look at their short working hour and high HR performance score
- Give them promotions so that they can increase the work efficiency of other groups, or give them more projects to handle
2. Cluster 1: This group of employees shows dedicated work ethic shown by their long working hour and also performance score
- Give them more flexible working hours and PTO in order to avoid causing attrition in this group caused by burnout
3. Cluster 2: This group of employees shows lacking work performance and have lower working hours
- Give them Performance Improvement Plan (PIP) with the sole purpose of having them work longer hours
4. Cluster 3: This group of employees shows the lowest level of efficiency if we take a look at their working hours and overtime, and also their low rating
- Give them PIP that act as a final warning with the sole purpose of increasing their work efficiency, but if they show no improvements, managers can lay them off
### 7 Conclusion
We have successfully created a Clustering Algorithm that can divide the group evenly. This model should be used as only suggestion and not decision maker for both HR teams and managers. For business insight and visualization regarding how the data looks like, you can check the EDA part in the deployment link above.

## Thank You For Visiting!!

📬 Connect with us:
Ayman Baswedan as Data Engineer
💼 Linkedin: https://www.linkedin.com/in/ayman-baswedan-b9a664326/ 

Bramantyo Anandaru as Data Analyst
💼 Linkedin: https://www.linkedin.com/in/bramantyo-anandaru-suyadi-0b9729208/ 

Reza M Syahfi as Data Scientist
💼 Linkedin: https://www.linkedin.com/in/reza-mrhafi/ 

Dini Anggriyani as Data Scientist
💼 Linkedin: https://www.linkedin.com/in/dini-a/ 