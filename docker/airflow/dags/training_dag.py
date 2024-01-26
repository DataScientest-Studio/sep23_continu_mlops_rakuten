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

training_dag = DAG(
   dag_id="training_script",
#    schedule_interval="@daily",
   schedule_interval=None,
   default_args={
        'start_date': days_ago(0),
    }
)


# tâche pour vérifier la présence de nouvelles données
def check_new_product( task_instance):
   df_new_texts = pd.read_csv('/app/drive/data/new_products.csv', usecols=['designation', 'description', 'productid', 'imageid'])
   return 'no_new_data_task' if len(df_new_texts) == 0 else 'branch_task2'
      
""" check_new_product_script = PythonOperator(
   task_id='check_new_product_script_task',
   python_callable=check_new_product,
   dag=training_dag
) """

branch_task = BranchPythonOperator(
    task_id='branch_task',
    python_callable=check_new_product,
    dag=training_dag
)
  

# tâches pour rajouter les textes et les images dans les données d'entrainement
def get_new_texts():
   
## on part d un df vide
   #df_x_train = pd.read_csv('/app/drive/data/X_train_update.csv', usecols=['designation', 'description', 'productid', 'imageid'])
   #df_y_train=pd.read_csv('/app/drive/data/Y_train.csv', usecols=['prdtypecode'])

   df_product_feedback = pd.read_csv('/app/drive/data/ProductUserFeedback.csv', usecols=['productid', 'categoryid'])
   df_new_product = pd.read_csv('/app/drive/data/new_products.csv', usecols=['designation', 'description', 'productid', 'imageid'])

   df_merged = pd.merge(df_new_product, df_product_feedback, on='productid')
   y_merged=df_merged['categoryid']
   y_merged = y_merged.to_frame(name='prdtypecode')
   df_merged=df_merged.drop('categoryid',axis=1)

   if df_merged.empty:
      return 'end_task'
   else:
      X_train =df_merged
            
      X_train.to_csv('/app/drive/data/X_train_update.csv', index=False)
      y_merged.to_csv('/app/drive/data/Y_train.csv', index=False)

      ###vider les fichiers
      df_product_feedback= pd.DataFrame(columns=df_product_feedback.columns)
      df_product_feedback.to_csv('/app/drive/data/ProductUserFeedback.csv', index=False)

      df_new_product= pd.DataFrame(columns=df_new_product.columns)
      df_new_product.to_csv('/app/drive/data/new_products.csv', index=False)

      

      for _, row in X_train.iterrows():
            image_filename = f"image_{row['imageid']}_product_{row['productid']}.jpg"
            source_path = os.path.join('/app/drive/images/new_images/', image_filename)
            destination_path = os.path.join('/app/drive/images//new_images/image_train/', image_filename)

            # Vérifier si le fichier existe avant de le déplacer
            if os.path.exists(source_path):
                # Déplacer l'image de source à destination
                shutil.move(source_path, destination_path)
      return 'run_training_script_task'




branch_task2 = BranchPythonOperator(
    task_id='branch_task2',
    python_callable=get_new_texts,
    dag=training_dag
)



# tâche pour lancer l'entrainement du modèle
run_training_script = BashOperator(
   task_id="run_training_script_task",
   bash_command="docker exec docker-entrainement-1 python training_quick.py",
   dag=training_dag
)



# tâche pour récupérer le meilleur modèle
def Get_best_model () : 
   mlflow.set_tracking_uri("http://entrainement:5000")
   #////////////////////////////////////////////////////////
   experiment_name = 'training_rakuten'
   current_experiment=dict(mlflow.get_experiment_by_name(experiment_name))
   experiment_id=current_experiment['experiment_id']
   all_experiments = [exp.experiment_id for exp in mlflow.search_experiments()]
   #///////////////////////////////////////////////////////
   # Récupérer l'accuracy du dernier entrainement 
   client = mlflow.tracking.MlflowClient()
   last_run = client.search_runs(experiment_ids="454740327196184364" ,filter_string="",run_view_type=ViewType.ACTIVE_ONLY,  order_by=["start_time DESC"], max_results=1,)
   last_run_accuracy = last_run[0].data.metrics['accuracy']
   # récuperer l'id du dernier entrainement qu'on utilisera pour stocker le modèle si performant
   last_run_id = last_run[0].info.run_id
   # Récupérer la meilleur accuracy de tous les entrainements sur Mlflow 
   Best_run = client.search_runs(experiment_ids="454740327196184364",filter_string="",run_view_type=ViewType.ACTIVE_ONLY,max_results=1,
    order_by=["metrics.accuracy DESC"],
    )[0]
   Best_run_accuracy =  Best_run.data.metrics['accuracy']
   # Vérifier si le dernier entrainement est le meilleur entrainement
   if last_run_accuracy >= 0.7 : 
        # load et enregister le modèle dans le cas d'une meilleur accuracy
        model = mlflow.sklearn.load_model("runs:/" + last_run_id + "/model")
        model_path=os.path.join(path_model_prod,'bimodal.h5')
        model.save(model_path, include_optimizer=False)
        return last_run_accuracy
   else : 
       #sinon retourner la meilleur accuracy déja connu
       return Best_run_accuracy
   
def function_with_return(task_instance):   
   # Récuperer la meilleur accuracy 
   best_accuracy =  Get_best_model()
   task_instance.xcom_push(
      key="Meilleur_accuracy",
      value= best_accuracy
    )
   
accuracy_task = PythonOperator(
    task_id='Get_accuracy_task',
    dag=training_dag,
    python_callable=function_with_return
)



no_new_data_task = DummyOperator(
    task_id='no_new_data_task',
    dag=training_dag,
)
end_task = DummyOperator(
    task_id='end_task',
    dag=training_dag,
)

# dépendances
#check_new_product_script >> branch_task
branch_task >> [no_new_data_task, branch_task2]
branch_task2 >> [end_task, run_training_script]
run_training_script >> accuracy_task >> end_task

