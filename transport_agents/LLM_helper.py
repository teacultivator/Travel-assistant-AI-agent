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

#     prompt = f"""
#     You are a helpful Flight Searching Agent. Your task is to satisfy client's requirements.

#     The user asked: "{user_query}"

#     Here is the raw data of flights offers (Amadeus API response):
#     {raw_json}

#     You must follow these steps IN ORDER:

#     Step 1: From the raw flight offers, SELECT ONLY those flights that satisfy ALL of the user’s constraints. 
#         - Constraints come directly from the user query.
#         - If a flight does not satisfy ALL constraints, IGNORE it completely.
        
#     Step 2: 
#     2. For each relevant flight, ALWAYS extract the fields mentioned below (mandatory). Only extract the following fields:
#        - airline
#        - price (total, in Indian Rupees)
#        - duration
#        - departure_time
#        - arrival_time
#        - stops (integer number of stops)
       

#     Step 3: Produce the final output as valid JSON with this exact structure:
#    {{
#     "summary": "short human-friendly text summarizing results in 1-2 sentences",
#     "filtered_results": [
#         {{
#         "airline": "...",
#         "price": "...",
#         "duration": "...",
#         "departure_time": "...",
#         "arrival_time": "...",
#         "stops": "..."
#         }}
#     ]
#    }}
#     You may use raw_json["data"][<index of offer>]["price"]["total"] to extract the price.
#     VERY IMPORTANT:
#         - Do not include flights that fail constraints.
#         - Do not include extra text or formatting outside JSON.
#         - If no flights meet the criteria, return:
#         {{"summary": "No flights found", "filtered_results": []}}
#     """

    def safe_json_parse(raw_text: str):
        # Remove markdown fences if present
        clean = re.sub(r"^```json\s*|\s*```$", "", raw_text.strip(), flags=re.DOTALL)
        return json.loads(clean)
    
    prompt0 = f"""
    You are a data filtering agent used by a flights booking agency. Given a user query, you need to generate the filter condition matching user query text.
    You are a JSON-only extractor. Return a *single JSON object* and nothing else, with no surrounding markdown.
    User query: {user_query}
    """

    response0 = model.generate_content(prompt0)
    print(response0.text)

    prompt = f"""
    You are a data extracter.
    
    Here is the raw JSON data of 40 flight offers:
    {raw_json}

    Filter Condition: {response0.text}

    You are given a dataset. Only extract results that match the required criteria. If no exact matches are found, respond with:
    {{"summary": NO results found.", filtered_results": []}}

    Follow these steps with precision:

    Step 1: Iterate and Evaluate Each Flight:
    - Go through every single flight offer in the raw_json data, one by one.
    - For each flight, compare its details against ALL of the criterias in filter condition.
    - A flight offer is only a match if it satisfies EVERY SINGLE CRITERION PERFECTLY.
    - If a flight offer fails even one criterion, you must discard it immediately and move to the next. There are no partial matches.

    Step 2: Extract and Format the Final Output:
    - For the flights that passed the strict evaluation in Step 2, extract ONLY the following fields:
    - `airline`
    - `price` (the total price, in Indian Rupees)
    - `duration`
    - `departure_time`
    - `arrival_time`
    - `stops` (the integer number of stops)
    - Construct the final output as a single, valid JSON object with the exact structure below, and return it.

    Final JSON Structure:
    {{
    "summary": "A brief, human-friendly 1-2 lines summary of the results (e.g., 'I found 3 late-night flights with exactly 2 stops.').",
    "filtered_results": [
        {{
        "airline": "...",
        "price": "...",
        "duration": "...",
        "departure_time": "...",
        "arrival_time": "...",
        "stops": ... 
        }}
    ]
    }}

    You may use raw_json["data"][<index of offer>]["price"]["total"] to extract the price.
    """

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
