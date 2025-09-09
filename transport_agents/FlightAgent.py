import pandas as pd
from langgraph.graph import StateGraph, END
from typing import TypedDict

# Define the agent's "state"
class FlightState(TypedDict):
    """State container for the Flight Search Agent."""
    source: str
    destination: str
    date: str
    results: list

# Load CSV dataset
flights_df = pd.read_csv("flights1.csv")

# Flight search function
def search_flights(state: FlightState) -> FlightState:
    # identifying requirements from incoming json file
    source = state.get("source", "").lower()
    destination = state.get("destination", "").lower()
    date = state.get("date", "")

    # When we integrate the API into this, we will also receive data about different booking services for the same flight.
    # Hence, we will also need to add those parameters into the state object.
    try:
        matches = flights_df[
            (flights_df["origin"].str.lower() == source) &
            (flights_df["destination"].str.lower() == destination) &
            (flights_df["departure_date"] == date)
        ]

        # Sort by price for now
        matches = matches.sort_values(by="price_usd")

        state["results"] = matches.to_dict(orient="records")
        print(state)
    except Exception:
        state["results"] = None
    return state

# Build the LangGraph workflow
graph = StateGraph(FlightState)

graph.add_node("search_flights", search_flights)
graph.set_entry_point("search_flights")
graph.add_edge("search_flights", END)

# Compile graph
app = graph.compile()

# Example query
if (__name__ == "__main__"):
    # sample user query
    user_query = {
        "source": "New York",
        "destination": "London",
        "date": "2025-09-08"
    }

    final_state = app.invoke(user_query)
    print("Available Flights:")
    try:
        for flight in final_state["results"]:
            print(f"\n{flight['airline']} | "
                f"{flight['departure_airport']} -> {flight['arrival_airport']} | " f"Departure Time:{flight['departure_time']}  Arrival Time:{flight['arrival_time']} | " f"Duration: {flight['duration']} | Cost: ${flight['price_usd']} | Class: {flight['class']} | Stops: {flight['stops']}")
    except Exception:
        print("No flights available for the specified details.")