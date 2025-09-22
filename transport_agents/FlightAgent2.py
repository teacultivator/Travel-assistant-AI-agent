from langgraph.graph import StateGraph, END  
from transport_agents.API_helper import get_access_token, search_flights
from graph.state import State
from transport_agents.LLM_helper import filter_and_extract_flights, print_flights_table

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

    # state["next_agent"] = "Result_Analysis_Agent"
    state["next_agent"] = "end"
    return state



# Test run
if __name__ == "__main__":
    # Build workflow
    graph = StateGraph(State)
    graph.add_node("flight_search_node", flight_search_node)
    graph.set_entry_point("flight_search_node")
    graph.add_edge("flight_search_node", END)

    flightSearchAgent = graph.compile()
    user_query = "Find me quick flights from Pune to Delhi on next Tuesday?"
    initial_state = {
        "origin": "Pune",
        "destination": "Delhi",
        "departure_date": "2025-09-30",
        "flight_results": {},
        "user_query": user_query,  # keep query in state so node can use it
    }

    print("Running flight search agent...")
    final_state = flightSearchAgent.invoke(initial_state)
