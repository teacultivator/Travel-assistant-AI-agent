from typing import TypedDict, List, Dict
import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("AMADEUS_API_KEY")
API_SECRET = os.getenv("AMADEUS_API_SECRET")

def get_access_token():
    url = "https://test.api.amadeus.com/v1/security/oauth2/token"
    response = requests.post(
        url,
        data={
        "grant_type": "client_credentials",
        "client_id": API_KEY,
        "client_secret": API_SECRET
    })
    if (response.status_code != 200):
        raise Exception(f"Failed to generate access token. Status code: {response.status_code}\n{response.text}")
    token = response.json()["access_token"]
    return token

def search_flights(origin, destination, date, token):
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