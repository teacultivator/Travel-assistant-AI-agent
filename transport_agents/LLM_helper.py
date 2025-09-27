from tabulate import tabulate
import google.generativeai as genai
import os, json, re
from dotenv import load_dotenv

def print_flights_table(flight_results):
    if (not flight_results):
        print("No flights to display.")
        return
    
    table = []
    for f in flight_results:
        table.append([
            f.get("airline", ""),
            f.get("price", ""),
            f.get("duration", ""),
            f.get("departure_time", ""),
            f.get("arrival_time", ""),
            f.get("stops", "")
        ])
    
    headers = ["Airline", "Price (USD)", "Duration", "Departure", "Arrival", "Stops"]
    print(tabulate(table, headers=headers, tablefmt="fancy_grid"))

load_dotenv()

genai.configure(api_key = os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")


def filter_and_extract_flights(user_query: str, raw_results: dict):
    """
    Sends the raw Amadeus API results + user query to Gemini.
    Gemini filters relevant flights, extracts fields into JSON, and summarizes.
    """
    raw_json = json.dumps(raw_results, indent=2)

    prompt = f"""
    You are a helpful Flight Searching Agent. Your task is to satisfy client's requirements.

    The user asked: "{user_query}"

    Here is the raw data of flights offers (Amadeus API response):
    {raw_json}

    You must follow these steps IN ORDER:

    Step 1: From the raw flight offers, SELECT ONLY those flights that satisfy ALL of the user’s constraints. 
        - Constraints come directly from the user query.
        - If a flight does not satisfy ALL constraints, IGNORE it completely.
        
    Step 2: 
    2. For each relevant flight, ALWAYS extract the fields mentioned below (mandatory). Only extract the following fields:
       - airline
       - price (total, in Indian Rupees)
       - duration
       - departure_time
       - arrival_time
       - stops (integer number of stops)
       

    Step 3: Produce the final output as valid JSON with this exact structure:
   {{
    "summary": "short human-friendly text summarizing results in 1-2 sentences",
    "filtered_results": [
        {{
        "airline": "...",
        "price": "...",
        "duration": "...",
        "departure_time": "...",
        "arrival_time": "...",
        "stops": "..."
        }}
    ]
   }}
    You may use raw_json["data"][<index of offer>]["price"]["total"] to extract the price.
    VERY IMPORTANT:
        - Do not include flights that fail constraints.
        - Do not include extra text or formatting outside JSON.
        - If no flights meet the criteria, return:
        {{"summary": "No flights found", "filtered_results": []}}
    """

    def safe_json_parse(raw_text: str):
        # Remove markdown fences if present
        clean = re.sub(r"^```json\s*|\s*```$", "", raw_text.strip(), flags=re.DOTALL)
        return json.loads(clean)

    response = model.generate_content(prompt)
    print(response.text)
    try:
        parsed = safe_json_parse(response.text)
        return parsed
    except Exception:
        return {
            "summary": "Could not parse Gemini output. Showing no filtered results.",
            "filtered_results": []
        }
