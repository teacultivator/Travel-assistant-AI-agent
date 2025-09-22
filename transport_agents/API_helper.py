import os, requests, time, re
from datetime import datetime, date
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

API_KEY = os.getenv("AMADEUS_API_KEY")
API_SECRET = os.getenv("AMADEUS_API_SECRET")

ACCESS_TOKEN = None
TOKEN_EXPIRY = 0

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AIRPORTS_FILE = os.path.join(BASE_DIR, "Airports1.csv")

COMMON_AIRPORTS = {
    'paris': 'CDG',
    'london': 'LHR',
    'new york': 'JFK',
    'los angeles': 'LAX',
    'tokyo': 'NRT',
    'madrid': 'MAD',
    'barcelona': 'BCN',
    'rome': 'FCO',
    'amsterdam': 'AMS',
    'frankfurt': 'FRA',
    'munich': 'MUC',
    'berlin': 'BER',
    'vienna': 'VIE',
    'zurich': 'ZUR',
    'geneva': 'GVA'
}

try:
    _airports_df = pd.read_csv(AIRPORTS_FILE)
    if 'City' in _airports_df.columns and 'IATA_Code' in _airports_df.columns:
        _airports_df = _airports_df[["City", "IATA_Code"]].dropna()
    elif 'city' in _airports_df.columns and 'iata_code' in _airports_df.columns:
        _airports_df = _airports_df[["city", "iata_code"]].dropna()
        _airports_df = _airports_df.rename(columns={"city": "City", "iata_code": "IATA_Code"})
    elif 'CITY' in _airports_df.columns and 'IATA' in _airports_df.columns:
        _airports_df = _airports_df[["CITY", "IATA"]].dropna()
        _airports_df = _airports_df.rename(columns={"CITY": "City", "IATA": "IATA_Code"})
    
    _airports_df["City"] = _airports_df["City"].str.strip().str.lower()
    print(f"Loaded {len(_airports_df)} airport codes from CSV")
    
except Exception as e:
    print(f"Error loading airports CSV: {e}")
    _airports_df = pd.DataFrame(columns=["City", "IATA_Code"])

def validate_date(date_str: str) -> tuple[bool, str]:
    try:
        flight_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        today = date.today()
        
        if flight_date < today:
            days_diff = (today - flight_date).days
            return False, f"Date {date_str} is {days_diff} days in the past"
        
        max_days_ahead = 330
        if (flight_date - today).days > max_days_ahead:
            return False, f"Date {date_str} is too far in the future"
            
        return True, ""
        
    except ValueError as e:
        return False, f"Invalid date format '{date_str}'"

def get_access_token():
    global ACCESS_TOKEN, TOKEN_EXPIRY
        
    if ACCESS_TOKEN and time.time() < TOKEN_EXPIRY:
        return ACCESS_TOKEN
 
    url = "https://test.api.amadeus.com/v1/security/oauth2/token"
    data = {
        "grant_type": "client_credentials",
        "client_id": API_KEY,
        "client_secret": API_SECRET
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
        
    try:
        response = requests.post(url, data=data, headers=headers, timeout=10)
        if response.status_code != 200:
            raise Exception(f"Failed to generate access token: {response.status_code}")
        
        data = response.json()
        ACCESS_TOKEN = data["access_token"]
        TOKEN_EXPIRY = time.time() + int(data["expires_in"]) - 20
        
        return ACCESS_TOKEN
        
    except requests.exceptions.RequestException as e:
        raise Exception(f"Network error while getting access token: {e}")

def _get_iata_from_city(city: str) -> str:
    if not city:
        raise ValueError("City name is required")
    
    city_norm = city.strip().lower()
    
    exact_match = _airports_df[_airports_df["City"].str.lower() == city_norm]
    if not exact_match.empty:
        return exact_match.iloc[0]["IATA_Code"]
    
    partial_match = _airports_df[_airports_df["City"].str.contains(city_norm, na=False)]
    if not partial_match.empty:
        return partial_match.iloc[0]["IATA_Code"]
    
    if city_norm in COMMON_AIRPORTS:
        return COMMON_AIRPORTS[city_norm]
    
    available_cities = _airports_df["City"].str.title().tolist()[:15]
    raise ValueError(f"No IATA code found for city: '{city}'")

def _is_iata_code(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Z]{3}", value or ""))

def search_flights(origin_city: str, destination_city: str, date: str, token: str):
    try:
        is_valid_date, date_error = validate_date(date)
        if not is_valid_date:
            return {
                "error": "INVALID_DATE",
                "message": date_error,
                "data": []
            }
        
        try:
            origin_code = origin_city if _is_iata_code(origin_city) else _get_iata_from_city(origin_city)
            destination_code = destination_city if _is_iata_code(destination_city) else _get_iata_from_city(destination_city)
        except ValueError as e:
            return {
                "error": "IATA_LOOKUP_ERROR", 
                "message": str(e),
                "data": []
            }
        
        url = "https://test.api.amadeus.com/v2/shopping/flight-offers"
        params = {
            "originLocationCode": origin_code,
            "destinationLocationCode": destination_code,
            "departureDate": date,
            "adults": 1,
            "max": 40
        }
        headers = {"Authorization": f"Bearer {token}"}
        
        response = requests.get(url, params=params, headers=headers, timeout=70)  
        result = response.json()
        
        if response.status_code != 200:
            error_msg = "Unknown API error"
            if "errors" in result and result["errors"]:
                error = result["errors"][0]
                error_msg = f"{error.get('title', '')}: {error.get('detail', '')}"
            
            return {
                "error": f"API_ERROR_{response.status_code}",
                "message": error_msg,
                "data": []
            }
            
        return result
        
    except Exception as e:
        return {
            "error": "SEARCH_ERROR",
            "message": str(e),
            "data": []
        }