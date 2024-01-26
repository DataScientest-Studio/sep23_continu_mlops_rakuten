from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.python_operator import PythonOperator
import requests
import os
 
"""
Ce DAG contient 2 tests de l'API :

    - Un test d'authentification. Les credentials sont sauvée dans le .env du docker compose
    - Un test de vérification de prediction. Si la prédiction est mauvaise, il faudra vérifier le modèle. La réussite de la prédiction indique également que le docker rclone est OK.
    - une notification par email pourrait être envisagée

"""

test_DAG = DAG(
   dag_id="test_API",
   schedule_interval="@daily",
   catchup=False,
   default_args={
        'start_date': days_ago(0),
       # 'email': ['mlops@rakuten.com'],  
       # 'email_on_failure': True,
       # 'email_on_retry': False,
       # 'retries': 1,
    }

)


BASE_URL = "http://api:8000"  

#### Reference de la réponse attendue pour l id 516376098
reponse_attendue = {
    "product id": 516376098,
    "product designation": "folkmanis puppets 1111 marionnette theatre mini turtle",
    "product description": "nodata",
    "product image": "image_1019294171_product_516376098.jpg",
    "predicted category": "Jouets et peluches enfant"
}

def authentification_test():
    
    credentials = {
    "username": os.environ.get("API_USERNAME"),
    "password": os.environ.get("API_PASSWORD")
    }
 
    # Obtenir un token JWT
    response = requests.post(f"{BASE_URL}/token", data=credentials)
    if response.status_code == 200:
        token_data = response.json()
        jwt_token = token_data['access_token']
        print("Token obtenu :", jwt_token)
    else:
        print("Erreur lors de l'obtention du token :", response.status_code, response.text)
        exit()

    # Utiliser le token JWT
    headers = {"Authorization": f"Bearer {jwt_token}"}
    secured_endpoint = "/secured"  
    response = requests.get(f"{BASE_URL}{secured_endpoint}", headers=headers)

    if response.status_code == 200:
        print("Réponse du point de terminaison sécurisé :", response.json())
    else:
        raise Exception(f"Erreur lors de l'accès au point de terminaison sécurisé : {response.status_code}, {response.text}")


def test_predictcategory():
  
    # Requête au point de terminaison predictcategory
    response = requests.get(f"{BASE_URL}/predictcategory/516376098")

    if response.status_code == 200:
        response_data = response.json()
        if response_data != reponse_attendue:
            raise ValueError("La réponse obtenue ne correspond pas à la réponse attendue.")
        print("Réponse de predictcategory vérifiée et correcte :", response_data)
    else:
        raise Exception(f"Erreur lors de l'accès à predictcategory : {response.status_code}, {response.text}")

run_auth_test = PythonOperator(
   task_id="run_authentification_test",
   python_callable=authentification_test,
   dag=test_DAG
) 

run_predictcategory_test = PythonOperator(
    task_id="run_predictcategory_test",
    python_callable=test_predictcategory,
    dag=test_DAG
)

run_auth_test >> run_predictcategory_test