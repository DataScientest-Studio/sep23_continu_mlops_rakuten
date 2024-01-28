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
import glob
import shutil
import tarfile

path_model_prod=  r'/app/drive/models/'  

training_dag = DAG(
   dag_id="training_script2",
#    schedule_interval="@daily",
   schedule_interval=None,
   default_args={
        'start_date': days_ago(0),
    }
)


# Fonction pour vérifier la présence d'un fichier .tar. Si le fichier tar est présent, le fichier est décompresser dans le dossier donnees_entrainement
def check_tar_files():
    file_path = '/app/drive/donnees_entrainement/data.tar'
    extract_path='/app/drive/donnees_entrainement'
    
    # Vérifier si le fichier spécifique existe
    if os.path.exists(file_path):
        print(f"Fichier trouvé : {file_path}")
        with tarfile.open(file_path) as tar:
            tar.extractall(path=extract_path)
            print(f"File extracted to {extract_path}")
        return 'run_training_script_task'
    else:
        print("Fichier data.tar non trouvé")
        return 'end_task'
     


branch_task = BranchPythonOperator(
    task_id='branch_task',
    python_callable=check_tar_files,
    dag=training_dag
)
  

# tâche pour lancer l'entrainement du modèle
run_training_script = BashOperator(
   task_id="run_training_script_task",
   bash_command="docker exec docker-entrainement-1 python training_quick.py",
   dag=training_dag
)

### Si la tache d entrainement a fonctionné, les dossiers image_train, image_test sont deleted. On garde le fichier .tar pour la démo. En production, il faudrait l effacer et enlever le exclude ds rclone pour le fichier tar
cleanup_task = BashOperator(
    task_id='cleanup_directory',
    bash_command='rm -rf /app/drive/donnees_entrainement/images_train /app/drive/donnees_entrainement/images_test',
  ###  bash_command='rm -rf /app/drive/donnees_entrainement/images_train && rm -rf /app/drive/donnees_entrainement/images_test && rm -rf /app/drive/donnees_entrainement/data.tar',  on efface pas le fichier .tar pour la demo. Ca prendrait trop de temps a telecharger
    dag=training_dag,
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
   last_run_accuracy = last_run[0].data.params['accuracy']
   # récuperer l'id du dernier entrainement qu'on utilisera pour stocker le modèle si performant
   last_run_id = last_run[0].info.run_id
   # Récupérer la meilleur accuracy de tous les entrainements sur Mlflow 
   Best_run = client.search_runs(experiment_ids="454740327196184364",filter_string="",run_view_type=ViewType.ACTIVE_ONLY,max_results=1, order_by=["params.accuracy DESC"])[0]
   Best_run_accuracy =  Best_run.data.params['accuracy']
   
   last_run_accuracy = float(last_run_accuracy)
   Best_run_accuracy = float(Best_run_accuracy)

   # Vérifier si le dernier entrainement est le meilleur entrainement
   if last_run_accuracy >= 0.7 : 
        # Définir le chemin du fichier source et le chemin de destination
        chemin_source = '/app/drive/models_entrainement/bimodal.h5'
        chemin_destination = '/app/drive/models/bimodal.h5'
        try:
            shutil.copy(chemin_source, chemin_destination)
            print("Fichier copié avec succès.")
        except IOError as e:
            print("Impossible de copier le fichier. " + str(e))
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



end_task = DummyOperator(
    task_id='end_task',
    dag=training_dag,
)

# dépendances

branch_task >> [end_task, run_training_script]
run_training_script >> cleanup_task>> accuracy_task >> end_task

