from typing import TypedDict, List, Dict
import os, requests, time
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("AMADEUS_API_KEY")
API_SECRET = os.getenv("AMADEUS_API_SECRET")

# def get_access_token():
#     url = "https://test.api.amadeus.com/v1/security/oauth2/token"
#     response = requests.post(
#         url,
#         data={
#         "grant_type": "client_credentials",
#         "client_id": API_KEY,
#         "client_secret": API_SECRET
#     })
#     if (response.status_code != 200):
#         raise Exception(f"Failed to generate access token. Status code: {response.status_code}\n{response.text}")
#     token = response.json()["access_token"]
#     return token

ACCESS_TOKEN = None
TOKEN_EXPIRY = 0  # unix timestamp

def get_access_token():
    global ACCESS_TOKEN, TOKEN_EXPIRY
    
    # Reusing token if applicable
    if (ACCESS_TOKEN and time.time() < TOKEN_EXPIRY):
        return ACCESS_TOKEN

    # Otherwise, fetch a new token
    url = "https://test.api.amadeus.com/v1/security/oauth2/token"
    data = {
        "grant_type": "client_credentials",
        "client_id": API_KEY,
        "client_secret": API_SECRET
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    
    response = requests.post(url, data=data, headers=headers)
    if (response.status_code != 200):
        raise Exception(f"Failed to generate access token. Status code: {response.status_code}\n{response.text}")
    data = response.json()
    
    # Save new token and expiry into global variables
    ACCESS_TOKEN = data["access_token"]
    TOKEN_EXPIRY = time.time() + int(data["expires_in"]) - 20 # 20 seconds buffer
    
    return ACCESS_TOKEN

def search_flights(origin:str , destination:str , date:str , token):
    # origin and destination are expected to be in IATA code format
    url = "https://test.api.amadeus.com/v2/shopping/flight-offers"
    params = {
        "originLocationCode": origin,
        "destinationLocationCode": destination,
        "departureDate": date,
        "adults": 1,
        "max": 5
    }
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(url, params=params, headers=headers)
    return response.json()