from langgraph.graph import StateGraph, END, MessagesState
from typing import TypedDict, List, Dict, Optional
from enum import Enum
from API_helper import get_access_token, search_flights

from LLM_helper import filter_and_summarize_flights, print_flights_table

class TransportMode(Enum):
    FLIGHT = "flight"
    BUS = "bus"
    TRAIN = "train"

# class FlightState(TypedDict):
#     origin: str            # source city/airport
#     destination: str       # destination city/airport
#     departure_date: str    # departure date (YYYY-MM-DD)
#     flight_results: List[Dict]  # results from API

class State(MessagesState):
    # Input
    next_agent: str
    origin: Optional[str]
    destination: Optional[str]
    departure_date: Optional[str]
    return_date: Optional[str]
    departure_time: Optional[str]
    return_time: Optional[str]
    mode: Optional[TransportMode]

    # Output / processing
    train_results: Optional[List[Dict]]
    bus_results: Optional[List[Dict]]
    flight_results: Optional[List[Dict]]
    booking_options: list
    selected_option: dict
    booking_confirmed: bool

def flight_search_node(state: State) -> State:
    try:
        token = get_access_token()
        # get results in json format
        results = search_flights(
            state["origin"], state["destination"], state["departure_date"], token
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
        state["flight_results"] = flights
    except Exception as e:  # data does not exist in results
        state["flight_results"] = []
        print("Error fetching flights:", e)
    return state

# Building the LangGraph workflow
graph = StateGraph(State)
graph.add_node("flight_search_node", flight_search_node)
graph.set_entry_point("flight_search_node")
graph.add_edge("flight_search_node", END)

# Compile graph
app = graph.compile()

# Testing
# if __name__ == "__main__":
#     # Example test run
#     initial_state = {
#         "origin": "DEL",          # Mumbai
#         "destination": "JFK",     # New York
#         "departure_date": "2025-12-17",     # YYYY-MM-DD
#         "flight_results": []
#     }

#     print("Running flight search agent...")
#     final_state = app.invoke(initial_state)

#     # FIX: use "flight_results", not "flights"
#     flight_results = final_state.get("flight_results", [])
#     if not flight_results:
#         print("No flights found.")
#     else:
#         print(f"Found {len(flight_results)} flights:\n")
#         for i, f in enumerate(flight_results, 1):
#             print(f"{i}. Airline: {f['airline']} | Price: USD {f['price']}", end=" || ")
#             print(f"Departure Time: {f['start_time']} | Arrival Time: {f['end_time']} | No. of stops: {f['stops']}")
            
if __name__ == "__main__":
    user_query = "Find me the cheapest flights from Delhi to New York on Dec 17"
    initial_state = {
        "origin": "DEL",
        "destination": "JFK",
        "departure_date": "2025-12-17",
        "flight_results": []
    }

    print("Running flight search agent...")
    final_state = app.invoke(initial_state)

    flight_results = final_state.get("flight_results", [])
    if (not flight_results):
        print("No flights found.")
    else:
        # Pass results + query to Gemini
        llm_output = filter_and_summarize_flights(user_query, flight_results)

        # Print summary
        print("\nGemini Summary:")
        print(llm_output["summary"])

        # Print filtered flights in table
        print("\nFiltered Flight Results:")
        print_flights_table(llm_output["filtered_results"])