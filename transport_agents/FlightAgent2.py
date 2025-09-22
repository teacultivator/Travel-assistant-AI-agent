from langgraph.graph import StateGraph, END  
from transport_agents.API_helper import get_access_token, search_flights
from graph import state  # self built state 
from transport_agents.LLM_helper import filter_and_extract_flights, print_flights_table

State = state.State

def flight_search_node(state: State) -> State:
    try:
        token = get_access_token()
        results = search_flights(
            state["origin"], state["destination"], state["departure_date"], token
        )
        
        if results:
            # Use the original user query if available, else fallback
            user_query = state.get("user_query", f"Find me flights from {state['origin']} to {state['destination']} on {state['departure_date']}")

            # Call LLM for filtering + extraction
            llm_output = filter_and_extract_flights(user_query, results)
            # Store processed results in state
            state["flight_results"] = llm_output.get("filtered_results", [])

            # Print summary + results table
            print("\nGemini Summary:")
            print(llm_output.get("summary", ""))
            print("\nFlight Results:")
            print_flights_table(state["flight_results"])
        else:
            print("No flights found.")

    except Exception as e:
        state["flight_results"] = {}
        print("Error fetching or processing flights:", e)

    state["next_agent"] = "Result_Analysis_Agent"
    return state

# Build workflow
graph = StateGraph(State)
graph.add_node("flight_search_node", flight_search_node)
graph.set_entry_point("flight_search_node")
graph.add_edge("flight_search_node", END)

flightSearchAgent = graph.compile()

# Test run
if __name__ == "__main__":
    user_query = "Find me afternoon flights from Mumbai to London on 2 October?"
    initial_state = {
        "origin": "Mumbai",
        "destination": "LHR",
        "departure_date": "2025-10-02",
        "flight_results": {},
        "user_query": user_query,  # keep query in state so node can use it
    }

    print("Running flight search agent...")
    final_state = flightSearchAgent.invoke(initial_state)
