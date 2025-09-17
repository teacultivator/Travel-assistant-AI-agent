from tabulate import tabulate
import google.generativeai as genai
import os, json
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
            f.get("start_time", ""),
            f.get("end_time", ""),
            f.get("stops", "")
        ])
    
    headers = ["Airline", "Price (USD)", "Duration", "Departure", "Arrival"]
    print(tabulate(table, headers=headers, tablefmt="fancy_grid"))


load_dotenv()

genai.configure(api_key = os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")

def filter_and_summarize_flights(user_query: str, flight_results: list):
    """
    Uses Gemini 2.5 Flash to filter, summarize, and return JSON results based on the query.
    """
    flights_json = json.dumps(flight_results, indent=2)

    prompt = f"""
    The user asked: "{user_query}"

    Here are the flight options (JSON):
    {flights_json}

    Task:
    1. (MOST IMPORTANT) Filter these flights to match the user's intent (e.g. cheapest, shortest duration, specific airline, non-stop, etc).
    2. Summarize the filtered results in 1–2 sentences.
    3. Your output should be only a JSON object with fields:
       - "summary": string
       - "filtered_results": list of flight dicts (MUST have the same schema as input)
    """

    response = model.generate_content(prompt)
    try:
        # Gemini might return JSON inside text, so we need to parse it
        parsed = json.loads(response.text)
        return parsed
    except Exception:
        # Fallback: return all flights if parsing fails
        return {
            "summary": "Sure, here are all the flight options tailored to your needs.", # Dummy response
            "filtered_results": flight_results
        }
