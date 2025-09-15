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
            flights.append({
                "airline": carrier,
                "price": price,
                "duration": duration
            })
        state["flights"] = flights
    except Exception as e: # data does not exist in results
        state["flights"] = []
        print("Error fetching flights:", e)
    return state

# Building the LangGraph workflow
graph = StateGraph(FlightState)
graph.add_node("search_flights", search_flights)
graph.set_entry_point("search_flights")
graph.add_edge("search_flights", END)

# Compile graph
app = graph.compile()