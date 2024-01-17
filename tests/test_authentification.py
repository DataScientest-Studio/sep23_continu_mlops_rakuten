from fastapi.testclient import TestClient

from passlib.context import CryptContext
from api_v2  import api
client = TestClient(api)
#from test_api import app


#pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
#client = TestClient(app)
def test_api():
    
    response = client.post("/token", data={"username": "luffy", "password": "123456789", "grant_type": "password"},
                           headers={"content-type": "application/x-www-form-urlencoded"})
    
    assert response.status_code == 200
    response_data = response.json()
    assert "access_token" in response_data
    assert response_data
    print (response_data )
    #assert "token_type" == "bearer"
    
    
    #response = client.get("/status")
    #message = response.json()
    #assert message
    #assert response.json() == {"username": "luffy", "password": "123456789"}