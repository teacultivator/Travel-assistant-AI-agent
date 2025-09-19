from langgraph.graph import StateGraph, END, MessagesState
from typing import TypedDict, List, Dict, Optional
from enum import Enum
from API_helper import get_access_token, search_flights

from LLM_helper import filter_and_extract_flights, print_flights_table

class TransportMode(Enum):
    FLIGHT = "flight"
    BUS = "bus"
    TRAIN = "train"

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
    flight_results: Optional[Dict]   # raw API results
    booking_options: list
    selected_option: dict
    booking_confirmed: bool

def flight_search_node(state: State) -> State:
    try:
        token = get_access_token()
        results = search_flights(
            state["origin"], state["destination"], state["departure_date"], token
        )
        # Store RAW results (do NOT extract fields here)
        state["flight_results"] = results
    except Exception as e:
        state["flight_results"] = {}
        print("Error fetching flights:", e)
    return state

# Build workflow
graph = StateGraph(State)
graph.add_node("flight_search_node", flight_search_node)
graph.set_entry_point("flight_search_node")
graph.add_edge("flight_search_node", END)

# Compile
flightSearchAgent = graph.compile()

# Test run
if __name__ == "__main__":
    user_query = "Find me flights from New York to Zurich on the next Wednesday?"
    initial_state = {
        "origin": "JFK",
        "destination": "ZRH",
        "departure_date": "2025-09-24",
        "flight_results": {}
    }

    print("Running flight search agent...")
    final_state = flightSearchAgent.invoke(initial_state)

    raw_results = final_state.get("flight_results", {})
    if (not raw_results):
        print("No flights found.")
    else:
        # LLM handles filtering + extraction
        llm_output = filter_and_extract_flights(user_query, raw_results)

        # Print summary
        print("\nGemini Summary:")
        print(llm_output["summary"])

        # Print results table
        print("\nFiltered Flight Results:")
        print_flights_table(llm_output["filtered_results"])
