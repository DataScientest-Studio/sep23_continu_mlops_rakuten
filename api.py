from fastapi import FastAPI

api = FastAPI()

@api.get('/')
def get_index():
   return 'hello world'

@api.get('/prediction')
def get_prediction():
    return {'predicted category': "category"}