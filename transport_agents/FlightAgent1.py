from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict
from API_helper import get_access_token, search_flights

class FlightState(TypedDict):
    origin: str            # source city/airport
    destination: str       # destination city/airport
    date: str              # departure date (YYYY-MM-DD)
    flights: List[Dict]    # results from API

def flight_search_node(state: FlightState) -> FlightState:
    try:
        token = get_access_token()
        # get results in json format
        results = search_flights(
            state["origin"], state["destination"], state["date"], token
        )
        # Extract clean results
        flights = []
        for offer in results.get("data", []):
            price = offer["price"]["total"]
            carrier = offer["itineraries"][0]["segments"][0]["carrierCode"]
            duration = offer["itineraries"][0]["duration"]
            start_time = offer["itineraries"][0]["segments"][0]["departure"]["at"]
            end_time = offer["itineraries"][0]["segments"][-1]["arrival"]["at"]
            flights.append({
                "airline": carrier,
                "price": price,
                "duration": duration,
                "start_time": start_time,
                "end_time": end_time
            })
        state["flights"] = flights
    except Exception as e: # data does not exist in results
        state["flights"] = []
        print("Error fetching flights:", e)
    return state

# Building the LangGraph workflow
graph = StateGraph(FlightState)
graph.add_node("flight_search_node", flight_search_node)
graph.set_entry_point("flight_search_node")
graph.add_edge("flight_search_node", END)

# Compile graph
app = graph.compile()

# Testing
if __name__ == "__main__":
    # Example test run
    initial_state = {
        "origin": "BOM",          # Mumbai
        "destination": "DEL",     # Delhi
        "date": "2025-09-20",     # YYYY-MM-DD
        "flights": []
    }

    print("Running flight search agent...")
    final_state = app.invoke(initial_state)

    flights = final_state.get("flights", [])
    if (not flights):
        print("No flights found.")
    else:
        print(f"Found {len(flights)} flights:\n")
        for i, f in enumerate(flights, 1):
            print(f"{i}. Airline: {f["airline"]} | Price: ${f["price"]}", end = " || ")
            try:
                print(f"Departure Time: {f["start_time"]} | Arrival Time: {f["end_time"]}")
            except Exception:
                print("Could not fetch exact timings.")
                  
