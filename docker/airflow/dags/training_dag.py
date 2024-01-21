from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.bash import BashOperator
# from airflow.contrib.operators.docker_exec import DockerExecOperator
 
training_dag = DAG(
   dag_id="training_script",
#    schedule_interval="@daily",
   schedule_interval=None,
   default_args={
        'start_date': days_ago(0),
    }
)
 
run_training_script = BashOperator(
   task_id="run_training_script_task",
   bash_command="docker exec docker-entrainement-1 python /app/training_quick.py",
   dag=training_dag
)
 
# run_training_script = DockerExecOperator(
#     task_id="execute_training_script",
#     docker_image="docker-entrainement-1",
#     command=["python", "./training_quick.py"],
#     dag=training_dag
# )
