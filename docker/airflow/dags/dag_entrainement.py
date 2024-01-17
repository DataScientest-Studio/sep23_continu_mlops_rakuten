from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 17),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'dag_entrainement',
    default_args=default_args,
    description='DAG pour lancer l entrainement de notre model',
    schedule_interval=timedelta(days=1),
)

docker_compose_up = BashOperator(
    task_id='docker_compose_up',
    bash_command='docker-compose -f /opt/docker/entrainement/docker-compose.yml up -d',
    dag=dag,
)

docker_compose_down = BashOperator(
    task_id='docker_compose_down',
    bash_command='docker-compose -f /opt/docker/entrainement/docker-compose.yml down',
    dag=dag,
)

docker_compose_up >> docker_compose_down