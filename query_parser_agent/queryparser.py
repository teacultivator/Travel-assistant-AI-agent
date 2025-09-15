import os
import json
import datetime
from typing import Any, Dict, TypedDict, Optional, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, END

# Simple State class for testing (replace with your actual State import)
class State(Dict[str, Any]):
    """Simple state class for testing"""
    pass

# Initialize LLM consistently
llm = init_chat_model("google_genai:gemini-2.0-flash")

def create_query_parser_chain():
    """Creates the query parser chain"""
    
    parser_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a travel assistant that extracts travel information from user queries.

Current travel information state:
- Origin: {origin}
- Destination: {destination}  
- Departure Date: {departure_date}
- Return Date: {return_date}
- Departure Time: {departure_time}
- Return Time: {return_time}
- Mode: {mode}

INSTRUCTIONS:
1. Extract NEW travel information from the user's query
2. Only update fields that have new information
3. Keep existing information if not mentioned in the query
4. Use standard formats: dates as YYYY-MM-DD, times as HH:MM

RESPONSE FORMAT:
Return ONLY the fields that need updating in this exact format:
ORIGIN: [city name]
DESTINATION: [city name]
DEPARTURE_DATE: [YYYY-MM-DD]
RETURN_DATE: [YYYY-MM-DD]
DEPARTURE_TIME: [HH:MM]
RETURN_TIME: [HH:MM]
MODE: [flight/bus/train]

If no new information is found, respond with: NO_CHANGES

Examples:
User: "I want to fly from New York to Paris on December 25th"
Response:
ORIGIN: New York
DESTINATION: Paris
DEPARTURE_DATE: 2024-12-25
MODE: flight

User: "Actually make that a train"
Response:
MODE: train
"""),
        ("human", "{query}")
    ])
    
    return parser_prompt | llm

def query_parser_agent(state: State) -> Dict[str, Any]:
    """
    Parses user queries to extract travel information and updates state.
    
    Args:
        state: Current state containing messages and travel info
        
    Returns:
        Updated state with new travel information and next agent decision
    """
    try:
        messages = state.get("messages", [])
        if messages is None:
            messages = []
        
        # Extract the latest user message content
        query = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage) and hasattr(msg, 'content'):
                query = msg.content
                break
            elif hasattr(msg, 'content') and isinstance(msg.content, str):
                query = msg.content
                break
        
        if not query:
            response_msg = "❓ I didn't receive a query. Please tell me about your travel plans."
            return {
                **state,  # Preserve all existing state
                "messages": messages + [AIMessage(content=response_msg)],
                "next_agent": "query_parser_agent"
            }
        
        # Get current state values with defaults
        current_state = {
            "origin": state.get("origin", ""),
            "destination": state.get("destination", ""),
            "departure_date": state.get("departure_date", ""),
            "return_date": state.get("return_date", ""),
            "departure_time": state.get("departure_time", ""),
            "return_time": state.get("return_time", ""),
            "mode": state.get("mode", "")
        }
        
        # Create and invoke the parsing chain
        chain = create_query_parser_chain()
        response = chain.invoke({
            "query": query,
            **current_state
        })
        
        # Extract response content safely
        if hasattr(response, 'content'):
            response_content = response.content
            # Handle case where content might be a list or other types
            if isinstance(response_content, str):
                response_text = response_content.strip()
            elif isinstance(response_content, list):
                response_text = str(response_content).strip()
            else:
                response_text = str(response_content).strip()
        else:
            response_text = str(response).strip()
        
        # Parse LLM response and update state
        updated_fields = {}
        
        if response_text != "NO_CHANGES":
            lines = [line.strip() for line in response_text.split('\n') if line.strip()]
            
            for line in lines:
                if ':' in line:
                    field, value = line.split(':', 1)
                    field_key = field.strip().lower()
                    field_value = value.strip()
                    
                    # Map field names and validate
                    field_mapping = {
                        "origin": "origin",
                        "destination": "destination", 
                        "departure_date": "departure_date",
                        "return_date": "return_date",
                        "departure_time": "departure_time",
                        "return_time": "return_time",
                        "mode": "mode"
                    }
                    
                    if field_key in field_mapping and field_value:
                        updated_fields[field_mapping[field_key]] = field_value
        
        # Merge updated fields with current state
        final_state = {**current_state, **updated_fields}
        
        # Validate required fields
        required_fields = ["origin", "destination", "departure_date", "mode"]
        missing_fields = []
        
        for field in required_fields:
            if not final_state.get(field):
                missing_fields.append(field)
        
        # Determine response and next agent
        if missing_fields:
            field_prompts = {
                "origin": "departure city",
                "destination": "arrival city", 
                "departure_date": "departure date",
                "mode": "travel mode (flight, bus, or train)"
            }
            
            missing_prompts: List[str] = [field_prompts.get(field, field) for field in missing_fields]
            missing_list: str = ", ".join(missing_prompts)
            
            response_msg = f"🔄 I need more information. Please provide your {missing_list}."
            next_agent = "query_parser_agent"
            
        else:
            # All required info collected
            summary = f"""✅ Great! Here's your travel information:
🛫 From: {final_state['origin']} → {final_state['destination']}
📅 Date: {final_state['departure_date']}
🚌 Mode: {final_state['mode'].title()}"""
            
            if final_state.get('return_date'):
                summary += f"\n🔄 Return: {final_state['return_date']}"
            if final_state.get('departure_time'):
                summary += f"\n⏰ Departure: {final_state['departure_time']}"
                
            response_msg = summary + "\n\nProceeding to find options..."
            
            # Route to appropriate booking agent
            mode = final_state.get("mode", "flight").lower()
            if "bus" in mode:
                next_agent = "bus_agent"
            elif "train" in mode:
                next_agent = "train_agent"
            else:
                next_agent = "flight_agent"
        
        # Add response message to state
        new_messages = messages + [AIMessage(content=response_msg)]
        
        # Return complete updated state - FIXED: Include all state, not just updates
        return {
            **state,  # Preserve any other state fields
            "messages": new_messages,
            "next_agent": next_agent,
            **final_state  # Include all travel information (current + updated)
        }
        
    except Exception as e:
        # Ensure messages is defined for error case
        messages = state.get("messages", []) or []
        error_msg = f"❌ Sorry, I encountered an error processing your request: {str(e)}"
        return {
            **state,  # Preserve existing state
            "messages": messages + [AIMessage(content=error_msg)],
            "next_agent": "query_parser_agent"
        }

def print_state_info(state: State, title: str):
    """Helper function to print current state information"""
    print(f"\n{'='*50}")
    print(f"{title}")
    print(f"{'='*50}")
    
    # Print travel info
    travel_fields = ["origin", "destination", "departure_date", "return_date", 
                    "departure_time", "return_time", "mode"]
    
    print("🧳 Travel Information:")
    for field in travel_fields:
        value = state.get(field, "")
        print(f"  {field.replace('_', ' ').title()}: {value or 'Not set'}")
    
    print(f"\n🤖 Next Agent: {state.get('next_agent', 'Not set')}")
    
    # Print last message
    messages = state.get("messages", [])
    if messages:
        last_msg = messages[-1]
        print(f"\n💬 Last Response:")
        print(f"  {last_msg.content}")

def test_query_parser():
    """Test the query parser agent with various scenarios"""
    
    print("🚀 Starting Query Parser Agent Tests")
    print("=" * 60)
    
    # Test Case 1: Complete travel information
    print("\n📝 TEST CASE 1: Complete travel information")
    state1 = State({
        "messages": [HumanMessage(content="I want to fly from New York to London on December 25th, 2024")]
    })
    
    result1 = query_parser_agent(state1)
    print_state_info(result1, "Result 1: Complete Information")
    
    # Test Case 2: Incomplete information requiring follow-up
    print("\n📝 TEST CASE 2: Incomplete information")
    state2 = State({
        "messages": [HumanMessage(content="I want to travel to Paris")]
    })
    
    result2 = query_parser_agent(state2)
    print_state_info(result2, "Result 2: Missing Information")
    
    # Test Case 3: Follow-up with missing information
    print("\n📝 TEST CASE 3: Follow-up with missing info")
    state3 = State({
        "messages": [
            HumanMessage(content="I want to travel to Paris"),
            AIMessage(content="I need more information..."),
            HumanMessage(content="From Mumbai on January 15th, 2025 by train")
        ],
        "destination": "Paris"  # Previous state
    })
    
    result3 = query_parser_agent(state3)
    print_state_info(result3, "Result 3: Follow-up Information")
    
    # Test Case 4: Modification of existing booking
    print("\n📝 TEST CASE 4: Modify existing booking")
    state4 = State({
        "messages": [
            HumanMessage(content="Actually, change the mode to bus and departure time to 10:30 AM")
        ],
        "origin": "Mumbai",
        "destination": "Paris", 
        "departure_date": "2025-01-15",
        "mode": "train"
    })
    
    result4 = query_parser_agent(state4)
    print_state_info(result4, "Result 4: Modified Booking")
    
    # Test Case 5: Round trip booking
    print("\n📝 TEST CASE 5: Round trip booking")
    state5 = State({
        "messages": [HumanMessage(content="Book a round trip flight from Delhi to Tokyo, departing March 10th and returning March 20th, 2025")]
    })
    
    result5 = query_parser_agent(state5)
    print_state_info(result5, "Result 5: Round Trip")
    
    # Test Case 6: Empty query
    print("\n📝 TEST CASE 6: Empty query")
    state6 = State({
        "messages": []
    })
    
    result6 = query_parser_agent(state6)
    print_state_info(result6, "Result 6: Empty Query")
    
    print(f"\n{'='*60}")
    print("✅ All tests completed!")
    print("🎯 Check the results above to verify the query parser is working correctly")

if __name__ == "__main__":
    # Set up environment variables if needed
    # os.environ["GOOGLE_API_KEY"] = "your-api-key-here"  # Uncomment and add your API key
    
    try:
        test_query_parser()
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        print("🔧 Make sure you have set your GOOGLE_API_KEY environment variable")
        print("🔧 Also ensure all required packages are installed:")
        print("   pip install langchain langchain-google-genai langgraph")