from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import BranchPythonOperator

import pandas as pd
from mlflow.entities import ViewType
import mlflow
import os
import shutil

path_model_prod=  r'/app/drive/models/'  

feedback_dag = DAG(
   dag_id="feedback_dag",
   schedule_interval="@daily",
   default_args={
        'start_date': days_ago(0),
    }
)


# tâche pour vérifier la présence de nouvelles données
def check_new_product( task_instance):
   df_new_texts = pd.read_csv('/app/drive/data/new_products.csv', usecols=['designation', 'description', 'productid', 'imageid'])
   return 'end_task' if len(df_new_texts) == 0 else 'get_texts_task'
      


branch_task = BranchPythonOperator(
    task_id='branch_task',
    python_callable=check_new_product,
    dag=feedback_dag
)
  

# tâches pour rajouter les textes et les images dans les données d'entrainement
def get_new_texts():
   
## on part d un df vide
   df_x_train = pd.read_csv('/app/drive/data_to_curate/X_train_tocurate.csv')
   
   df_product_feedback = pd.read_csv('/app/drive/data/ProductUserFeedback.csv', usecols=['productid', 'categoryid'])
   df_product_feedback=df_product_feedback.rename(columns={'categoryid':'prdtypecode'})

   df_new_product = pd.read_csv('/app/drive/data/new_products.csv', usecols=['designation', 'description', 'productid', 'imageid'])
   df_merged = pd.merge(df_new_product, df_product_feedback, on='productid')
   print(df_merged)

   if df_merged.empty:
      return 
   else:
      X_train =pd.concat([df_x_train,df_merged],ignore_index=True)
      print(X_train)
      X_train.to_csv('/app/drive/data_to_curate/X_train_tocurate.csv', index=False)
      
      ###vider les fichiers
      column_names = ['', 'productid', 'categoryid', 'datetime']
      df_product_feedback_empty = pd.DataFrame(columns=column_names)
      
      df_product_feedback_empty.to_csv('/app/drive/data/ProductUserFeedback.csv', index=False)
      
      column_names = ['', 'designation', 'description', 'productid','imageid']
      df_new_product= pd.DataFrame(columns=column_names)
      df_new_product.to_csv('/app/drive/data/new_products.csv', index=False)

      

      for _, row in X_train.iterrows():
            image_filename = f"image_{row['imageid']}_product_{row['productid']}.jpg"
            source_path = os.path.join('/app/drive/images/new_images/', image_filename)
            destination_path = os.path.join('/app/drive/data_to_curate/images/', image_filename)

            # Vérifier si le fichier existe avant de le déplacer
            if os.path.exists(source_path):
                # Déplacer l'image de source à destination
                shutil.move(source_path, destination_path)
      return 


get_texts_task = PythonOperator(
    task_id='get_texts_task',
    python_callable=get_new_texts,
    dag=feedback_dag
)




end_task = DummyOperator(
    task_id='end_task',
    dag=feedback_dag
)

# dépendances
branch_task >> [end_task, get_texts_task]
get_texts_task >> end_task

