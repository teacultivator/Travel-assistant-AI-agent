from langgraph.graph import StateGraph, END  
from transport_agents.API_helper import get_access_token, search_flights
from graph.state import State
from transport_agents.LLM_helper import filter_and_extract_flights, print_flights_table
from langchain_core.messages import AIMessage
from typing import Dict, Any

def flight_search_node(state: State) -> Dict[str, Any]:
    """
    Flight search agent that integrates with the main workflow.
    Searches for flights and returns results with proper state management.
    """
    messages = state.get("messages", [])
    
    try:
        print(f"\n Searching for flights from {state['origin']} to {state['destination']} on {state['departure_date']}...")
        
        # Get access token and search flights
        token = get_access_token()
        results = search_flights(
            state["origin"], state["destination"], state["departure_date"], token
        )
        
        if results:
            # Use the original user query if available, else create a fallback
            user_query = state.get("user_query", 
                f"Find me flights from {state['origin']} to {state['destination']} on {state['departure_date']}")

            # Call LLM for filtering + extraction
            llm_output = filter_and_extract_flights(user_query, results)
            
            # Store processed results in state
            flight_results = llm_output.get("filtered_results", [])
            
            # Print summary + results table to console
            print("\n✈️ Gemini Summary:")
            print(llm_output.get("summary", "Flight search completed"))
            print("\n📊 Flight Results:")
            print_flights_table(flight_results)
            
            # Create response message for the chat
            if flight_results:
                response_msg = f" Found {len(flight_results)} flights from {state['origin']} to {state['destination']} on {state['departure_date']}. Check the console for detailed flight information."
            else:
                response_msg = f" No suitable flights found from {state['origin']} to {state['destination']} on {state['departure_date']}."
            
            # Return updated state with results
            return {
                **state,
                "flight_results": flight_results,
                "messages": messages + [AIMessage(content=response_msg)],
                "next_agent": "end",
                "needs_user_input": False
            }
            
        else:
            response_msg = f" No flights found from {state['origin']} to {state['destination']} on {state['departure_date']}."
            print(f"\n{response_msg}")
            
            return {
                **state,
                "flight_results": [],
                "messages": messages + [AIMessage(content=response_msg)],
                "next_agent": "end",
                "needs_user_input": False
            }

    except Exception as e:
        error_msg = f" Error searching for flights: {str(e)}"
        print(f"\n{error_msg}")
        
        return {
            **state,
            "flight_results": [],
            "messages": messages + [AIMessage(content=error_msg)],
            "next_agent": "end",
            "needs_user_input": False
        }

# if __name__ == "__main__":
#     # Build workflow for testing
#     graph = StateGraph(State)
#     graph.add_node("flight_search_node", flight_search_node)
#     graph.set_entry_point("flight_search_node")
#     graph.add_edge("flight_search_node", END)

#     flightSearchAgent = graph.compile()
    
#     # Test data
#     user_query = "Find me quick flights from Pune to Delhi on next Tuesday?"
#     initial_state = {
#         "messages": [],  # Added messages for consistency
#         "origin": "Pune",
#         "destination": "Delhi",
#         "departure_date": "2025-09-30",
#         "flight_results": {},
#         "user_query": user_query,
#         "next_agent": "flight_search_node",
#         "needs_user_input": False
#     }

#     print("Running flight search agent...")
#     final_state = flightSearchAgent.invoke(initial_state)
    
#     print("\n" + "="*50)
#     print("FINAL STATE:")
#     print("="*50)
#     print(f"Flight results count: {len(final_state.get('flight_results', []))}")
#     print(f"Next agent: {final_state.get('next_agent')}")
#     print(f"Messages count: {len(final_state.get('messages', []))}")
#     if final_state.get('messages'):
#         print(f"Last message: {final_state['messages'][-1].content}")