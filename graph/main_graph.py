from langgraph.graph import StateGraph,END
from typing import TypedDict, Annotated, List, Literal, Dict, Any
from graph.state import State
from query_parser_agent.queryparser import query_parser
from transport_agents.FlightAgent2 import flight_search_node
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

# Mock flight agent for testing
# def flight_search_node(state: State) -> Dict[str, Any]:
#     """Mock flight agent for testing"""
#     messages = state.get("messages", [])
#     response_msg = f"Flight search initiated for {state.get('origin')} to {state.get('destination')} on {state.get('departure_date')}"
    
    # return {
    #     **state,
    #     "messages": messages + [AIMessage(content=response_msg)],
    #     "next_agent": "end",
    #     "needs_user_input": False
    # }

def bus_search_node(state: State) -> Dict[str, Any]:
    """Mock bus agent for testing"""
    messages = state.get("messages", [])
    response_msg = f"Bus search initiated for {state.get('origin')} to {state.get('destination')} on {state.get('departure_date')}"
    
    return {
        **state,
        "messages": messages + [AIMessage(content=response_msg)],
        "next_agent": "end",
        "needs_user_input": False
    }

def train_search_node(state: State) -> Dict[str, Any]:
    """Mock train agent for testing"""
    messages = state.get("messages", [])
    response_msg = f"Train search initiated for {state.get('origin')} to {state.get('destination')} on {state.get('departure_date')}"
    
    return {
        **state,
        "messages": messages + [AIMessage(content=response_msg)],
        "next_agent": "end",
        "needs_user_input": False
    }

def create_workflow():
    """Create and configure the workflow graph"""
    workflow = StateGraph(State)
    
    # Add nodes
    workflow.add_node("query_parser", query_parser)
    workflow.add_node("flight_agent", flight_search_node)
    workflow.add_node("bus_agent", bus_search_node)
    workflow.add_node("train_agent", train_search_node)
    
    # Set entry point
    workflow.set_entry_point("query_parser")
    
    def router(state: State) -> Literal["query_parser", "flight_agent", "bus_agent", "train_agent", "__end__"]:
        next_agent = state.get("next_agent")
        if next_agent == "end":
            return END
        if next_agent == "wait_for_input":
            return END  # Stop processing when we need user input
        if next_agent in ["query_parser", "flight_agent", "bus_agent", "train_agent"]:
            return next_agent
        return "query_parser"
    
    # Add conditional edges for all nodes
    for node in ["query_parser", "flight_agent", "bus_agent", "train_agent"]:
        workflow.add_conditional_edges(node, router, {
            "query_parser": "query_parser",
            "flight_agent": "flight_agent",
            "bus_agent": "bus_agent",
            "train_agent": "train_agent",
            END: END
        })
    
    return workflow.compile()

def print_travel_state(state: Dict[str, Any]):
    """Print current travel information in a formatted way"""
    print("\n" + "="*50)
    print("CURRENT TRAVEL INFORMATION:")
    print("="*50)
    print(f"Origin: {state.get('origin', 'Not set')}")
    print(f"Destination: {state.get('destination', 'Not set')}")
    print(f"Departure Date: {state.get('departure_date', 'Not set')}")
    print(f"Return Date: {state.get('return_date', 'Not set')}")
    print(f"Departure Time: {state.get('departure_time', 'Not set')}")
    print(f"Return Time: {state.get('return_time', 'Not set')}")
    print(f"Mode: {state.get('mode', 'Not set')}")
    print("="*50 + "\n")

def interactive_chat():
    """Run interactive chat session with the travel assistant"""
    print("Welcome to the Interactive Travel Assistant!")
    print("Type your travel queries and I'll help you plan your trip.")
    print("Type 'quit', 'exit', or 'bye' to end the session.")
    print("Type 'reset' to start over with a new trip.")
    print("Type 'status' to see current travel information.")
    print("-" * 60)
    
    # Create workflow
    graph = create_workflow()
    
    # Initialize state
    current_state = {
        "messages": [],
        "origin": "",
        "destination": "",
        "departure_date": "",
        "return_date": "",
        "departure_time": "",
        "return_time": "",
        "mode": "",
        "next_agent": "query_parser",
        "needs_user_input": True
    }
    
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            # Handle special commands
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\nThank you for using the Travel Assistant! Have a great trip!")
                break
            
            elif user_input.lower() == 'reset':
                current_state = {
                    "messages": [],
                    "origin": "",
                    "destination": "",
                    "departure_date": "",
                    "return_date": "",
                    "departure_time": "",
                    "return_time": "",
                    "mode": "",
                    "next_agent": "query_parser",
                    "needs_user_input": True
                }
                print("\nTrip information reset! Please tell me about your new travel plans.")
                continue
            
            elif user_input.lower() == 'status':
                print_travel_state(current_state)
                continue
            
            elif not user_input:
                print("Please enter your travel query or type 'help' for commands.")
                continue
            
            # Add user message to state
            current_state["messages"].append(HumanMessage(content=user_input))
            current_state["needs_user_input"] = False  # We just got user input
            current_state["next_agent"] = "query_parser"  # Reset to parser for new input
            
            # Process with workflow
            print("\nAssistant: Processing...")
            
            # Process the user input through the workflow
            response_state = graph.invoke(current_state)
            current_state = response_state
            
            # Get and display the assistant's response
            messages = current_state.get("messages", [])
            if messages:
                # Find the last assistant message
                last_assistant_msg = None
                for msg in reversed(messages):
                    if isinstance(msg, AIMessage):
                        last_assistant_msg = msg
                        break
                
                if last_assistant_msg:
                    print(f"Assistant: {last_assistant_msg.content}")
            
            # Check if we need to continue processing (for booking agents)
            # Don't continue if we're waiting for user input or if we've ended
            if (not current_state.get("needs_user_input", False) and 
                current_state.get("next_agent") not in ["query_parser", "end", "wait_for_input"]):
                
                max_iterations = 5
                iterations = 0
                
                while (not current_state.get("needs_user_input", False) and 
                       current_state.get("next_agent") not in ["query_parser", "end", "wait_for_input"] and 
                       iterations < max_iterations):
                    
                    response_state = graph.invoke(current_state)
                    current_state = response_state
                    iterations += 1
                    
                    # Display any new assistant messages
                    messages = current_state.get("messages", [])
                    if messages:
                        last_msg = messages[-1]
                        if isinstance(last_msg, AIMessage):
                            # Only show if it's a new message (simple check)
                            print(f"Assistant: {last_msg.content}")
            
            # Show travel status if booking is complete
            if current_state.get("next_agent") == "end":
                print("\nTravel search completed!")
                print_travel_state(current_state)
                
                # Ask if they want to start a new search
                restart = input("\nWould you like to plan another trip? (y/n): ").strip().lower()
                if restart in ['y', 'yes']:
                    current_state = {
                        "messages": [],
                        "origin": "",
                        "destination": "",
                        "departure_date": "",
                        "return_date": "",
                        "departure_time": "",
                        "return_time": "",
                        "mode": "",
                        "next_agent": "query_parser",
                        "needs_user_input": True
                    }
                    print("\nReady for your next trip! Tell me about your travel plans.")
                else:
                    print("\nThank you for using the Travel Assistant! Have a great trip!")
                    break
                    
        except KeyboardInterrupt:
            print("\n\nSession interrupted. Thank you for using the Travel Assistant!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
            print("Please try again or type 'reset' to start over.")

if __name__ == "__main__":
    interactive_chat()