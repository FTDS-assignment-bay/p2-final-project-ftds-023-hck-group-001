import pandas as pd 
import numpy as np 
from airflow import DAG 
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.operators.python import PythonOperator
from elasticsearch import Elasticsearch 
from datetime import datetime


# Default Parameters
default_args = {
    "owner": "Group_1_Finpro",
    'retry': None,
    'start_date': datetime(2025, 1, 7)
}

# Function fetch from postgre
def LoadSQL(**context):
    # Koneksi ke database
    source_db = PostgresHook(postgres_conn_id = 'postgres_airflow')
    source_conn = source_db.get_conn()

    # Read data dari sql
    data_raw = pd.read_sql('SELECT * FROM employees ', source_conn)

    # Simpan data dari db ke .csv
    path  = '/opt/airflow/dags/datanya.csv'
    data_raw.to_csv(path, index=False)

    # Export path ke data raw
    context['ti'].xcom_push(key= 'raw_data_path', value=path)

# Function Data cleaning
def cleaning(**context):
    ti = context['ti']

    # read data
    data_raw = pd.read_csv('/opt/airflow/dags/datanya.csv')

    # cleaning
    data_raw = data_raw.dropna()
    data_raw = data_raw.drop_duplicates()
    data_clean = data_raw

    # simpan data clean
    path = '/opt/airflow/dags/data_clean.csv'
    data_clean.to_csv(path, index=False)

    # export path data clean
    context['ti'].xcom_push(key='clean_data_path', value=path)

# Function Load ke elasticsearch
def load_to_es(**context):
    # ambil konteks task instance
    ti = context['ti']

    # load clean data 
    data_clean = pd.read_csv('/opt/airflow/dags/data_clean.csv')

    # buat koneksi ke elasticsearch
    es = Elasticsearch('http://elasticsearch:9200') 
    
    if not es.ping():
        print('CONNECTION FAILED')
    
    # load data nya, per baris/streaming
    for i, row in data_clean.iterrows():

        doc = row.to_json()
        
        res = es.index(index='finpro_group1', doc_type = 'doc', body = doc)

# define dag
with DAG(
    'Finpro_group1',
    description = 'DAG finpro',
    schedule_interval = '0 6-7 1 1 *', # Setiap 1 Januari jam 6:00-7:00
    default_args = default_args,
    catchup = False
) as dag:

    # task extract
    extract_data = PythonOperator(
        task_id = 'LoadSQL',
        python_callable = LoadSQL,
        provide_context = True
    )

    # task transform
    transform_data = PythonOperator(
        task_id = 'Cleaning',
        python_callable = cleaning,
        provide_context = True
    )

    # task Load
    load_data = PythonOperator(
        task_id = 'load_data',
        python_callable = load_to_es,
        provide_context = True
    )

extract_data >> transform_data 

transform_data >> load_data