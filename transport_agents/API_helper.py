import os, requests, time, re
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

API_KEY = os.getenv("AMADEUS_API_KEY")
API_SECRET = os.getenv("AMADEUS_API_SECRET")

ACCESS_TOKEN = None
TOKEN_EXPIRY = 0  # unix timestamp

# Get the directory where this script (API_helper.py) resides
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Construct full path to Airports2.csv
AIRPORTS_FILE = os.path.join(BASE_DIR, "Airports1.csv")

# Load airport codes safely
_airports_df = pd.read_csv(AIRPORTS_FILE)[["City", "IATA_Code"]].dropna()
_airports_df["City"] = _airports_df["City"].str.strip().str.lower()

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

def _get_iata_from_city(city: str) -> str:
    # Return IATA code for a given city name (case-insensitive)
    if not city:
        raise ValueError("City name is required for IATA lookup.")
    city_norm = city.strip().lower()
    row = _airports_df[_airports_df["City"].str.lower() == city_norm]
    if row.empty:
        raise ValueError(f"No IATA code found for city: {city}")
    return row.iloc[0]["IATA_Code"]

def _is_iata_code(value: str) -> bool:
    # Checking if origin and destination are already valid IATA codes
    return bool(re.fullmatch(r"[A-Z]{3}", value or ""))

def search_flights(origin_city: str, destination_city: str, date: str, token: str):
    # Looking up IATA codes for cities
    origin_code = origin_city if _is_iata_code(origin_city) else _get_iata_from_city(origin_city)
    destination_code = destination_city if _is_iata_code(destination_city) else _get_iata_from_city(destination_city)

    url = "https://test.api.amadeus.com/v2/shopping/flight-offers"
    params = {
        "originLocationCode": origin_code,
        "destinationLocationCode": destination_code,
        "departureDate": date,
        "adults": 1,
        "max": 40
    }
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(url, params=params, headers=headers).json()
    print(response)
    return response