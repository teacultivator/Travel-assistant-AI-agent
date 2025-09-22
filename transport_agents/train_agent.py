import os
import re
import dateparser
import requests
from datetime import datetime
from langchain.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from typing import TypedDict, List, Dict, Optional, Annotated, Any
from enum import Enum
from langgraph.graph.message import add_messages
from langgraph.graph import MessagesState
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
import sys
import os
# Add the parent directory to the Python path to import from graph module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from graph.state import State


# Load environment variables for API keys
load_dotenv()

llm = init_chat_model("google_genai:gemini-2.0-flash")





CITY_TO_CODE = {
    "delhi": "NDLS",
    "new delhi": ["NDLS", "ANVT", "DLI", "NZM", "DEE", "DSA"],
    "patna": "PNBE",
    "mumbai": "CSTM",
    "kolkata": "HWH",
    "chennai": "MAS",
    # Add more as needed
}

def fetch_trains_by_day(date_str: str, source: str, destination: str) -> str:
    try:
        # Try different date parsing approaches
        dt = dateparser.parse(date_str, settings={'PREFER_DATES_FROM': 'future'})
        if not dt:
            # Try with current date context
            from datetime import datetime, timedelta
            today = datetime.now()
            if 'tuesday' in date_str.lower():
                # Find next Tuesday
                days_ahead = 1 - today.weekday()  # Tuesday is day 1
                if days_ahead <= 0:  # Target day already happened this week
                    days_ahead += 7
                dt = today + timedelta(days_ahead)
            elif 'monday' in date_str.lower():
                days_ahead = 0 - today.weekday()
                if days_ahead <= 0:
                    days_ahead += 7
                dt = today + timedelta(days_ahead)
            # Add more day mappings as needed
            
        if not dt:
            return f"Could not parse date: {date_str}. Please use format like '2023-10-15' or specific dates."
            
        weekday_key = dt.strftime("%a").lower()[:3]
        
        api_key = os.getenv("RAPIDAPI_KEY")
        if not api_key:
            return "API key not found. Please check your .env file."
        
        headers = {
            'x-rapidapi-key': api_key,
            'x-rapidapi-host': "irctc1.p.rapidapi.com"
        }
        url = f"https://irctc1.p.rapidapi.com/api/v3/getLiveStation?fromStationCode={source}&toStationCode={destination}&hours=8"
        
        print(f"DEBUG: API URL: {url}")
        print(f"DEBUG: Headers: {headers}")
        print(f"DEBUG: Looking for weekday: {weekday_key}")
        
        response = requests.get(url, headers=headers, timeout=15)
        print(f"DEBUG: Response Status: {response.status_code}")
        
        response.raise_for_status()
        data = response.json()
        print(f"DEBUG: API Response: {data}")
        
        trains = data.get("data", [])
        print(f"DEBUG: Total trains found: {len(trains)}")
        
        if trains:
            print(f"DEBUG: First train example: {trains[0]}")
        
        trains_today = [
            t for t in trains if t.get("runDays", {}).get(weekday_key, False)
        ]
        print(f"DEBUG: Trains running on {weekday_key}: {len(trains_today)}")
        
        if not trains_today:
            return f"No trains found on {dt.strftime('%d-%m-%Y')} from {source} to {destination}."
        formatted = "\n".join(
            f"{t['trainName']} ({t['trainNumber']}) at {t['departureTime']}" for t in trains_today
        )
        return f"Available trains on {dt.strftime('%d-%m-%Y')} from {source} to {destination}:\n{formatted}"
    except Exception as e:
        return f"API error: {str(e)}"


def resolve_city_code(city):
    """Convert city name to station code"""
    code_data = CITY_TO_CODE.get(city.strip().lower())
    if not code_data:
        return city.strip().upper()
    if isinstance(code_data, list):
        return code_data[0]
    return code_data

@tool
def train_options_tool(date_str: str, source: str, destination: str) -> str:
    """
    Fetches available trains between two stations on a given date using the IRCTC RapidAPI.
    date_str: Date in any format (e.g., '2023-10-15', 'next Monday').
    source: Source city name or station code (e.g., 'Delhi', 'Patna', 'NDLS', 'PNBE').
    destination: Destination city name or station code (e.g., 'Delhi', 'Patna', 'NDLS', 'PNBE').
    """
    source_code = resolve_city_code(source)
    destination_code = resolve_city_code(destination)
    
    return fetch_trains_by_day(date_str, source_code, destination_code)


tools = [train_options_tool]
llm_with_tools = llm.bind_tools(tools)

def train_search_node(state: State) -> Dict[str, Any]:
    """
    Main train agent node that processes user queries and decides whether to use tools or provide direct responses.
    Compatible with main_graph.py.
    """
    messages = state.get("messages", [])
    
    # CIRCUIT BREAKER: Check if we just got a tool result - if so, format and end , very important to break the look of api calling.
    if messages:
        last_msg = messages[-1]
        # Check if the last message looks like a tool result
        if (hasattr(last_msg, 'content') and last_msg.content and 
            ('Available trains on' in str(last_msg.content) or 
             'No trains found' in str(last_msg.content) or 
             'API error:' in str(last_msg.content))):
            
            print("DEBUG: Tool result detected, formatting final response")
            tool_result = last_msg.content
            formatted_response = f"🚂 **Train Search Results**\n\n{tool_result}\n\nHave a great journey!"
            
            return {
                **state,
                "messages": messages + [AIMessage(content=formatted_response)],
                "next_agent": "end",
                "needs_user_input": False
            }
    
    # Get the user query from the latest message or from state
    user_query = ""
    if messages:
        # Extract from the latest message, but skip tool results
        for msg in reversed(messages):
            if hasattr(msg, 'content') and msg.content:
                # Skip if this looks like a tool result
                if ('Available trains on' in str(msg.content) or 
                    'No trains found' in str(msg.content) or 
                    'API error:' in str(msg.content)):
                    continue
                user_query = msg.content
                break
            elif isinstance(msg, dict) and 'content' in msg:
                user_query = msg['content']
                break
    elif state.get("user_query"):
        user_query = state["user_query"]
    
    # If no user query, create one from state info for tool usage
    if not user_query and state.get("origin") and state.get("destination") and state.get("departure_date"):
        user_query = f"Find me trains from {state['origin']} to {state['destination']} on {state['departure_date']}"
    
    if not user_query:
        print("No user query found in state.")
        return {
            **state,
            "messages": messages + [AIMessage(content="I need a query to help you with train information.")],
            "next_agent": "end",
            "needs_user_input": True
        }

    print(f"Processing User Query: {user_query}")

    # Create a system prompt for the LLM to decide whether to use tools
    system_prompt = f"""You are a helpful train travel assistant. You can search for train schedules using the train_options_tool when users ask about specific train routes and dates.

Current context from state:
- Origin: {state.get('origin', 'Not specified')}
- Destination: {state.get('destination', 'Not specified')}
- Date: {state.get('departure_date', 'Not specified')}

Use the train_options_tool when:
1. User asks for trains between specific cities/stations
2. User mentions a specific date for travel
3. User wants to check train availability
4. You have enough information to make a search

If you have origin, destination, and date information (either from the query or context), use the tool to get actual train data.

Provide direct answers for:
1. General train travel advice
2. Information about train stations
3. Questions that don't require specific schedule lookup

When using the tool, extract the origin, destination, and date from the user's query or use the context information.
Be helpful and conversational in your responses."""

    try:
        # Prepare messages for LLM
        llm_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]

        print("DEBUG: Calling LLM with tools...")
        # Get response from LLM with tools
        response = llm_with_tools.invoke(llm_messages)
        print(f"DEBUG: LLM Response received: {type(response)}")
        
        # Return messages in the same format as working main.py
        return {"messages": [response]}
        
    except Exception as e:
        error_msg = f"Error processing train request: {str(e)}"
        print(f"Error in train_search_node: {error_msg}")
        return {
            **state,
            "messages": messages + [AIMessage(content=error_msg)],
            "next_agent": "end",
            "needs_user_input": False
        }



tool_node = ToolNode(tools)

# Build the main graph
graph = StateGraph(State)
graph.add_node("train_search_node", train_search_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "train_search_node")
graph.add_conditional_edges("train_search_node", tools_condition)
graph.add_edge("tools", "train_search_node")  # Go back to format the tool result

train_chatbot = graph.compile()

# Test run
if __name__ == "__main__":
    # Predefined user query test
    user_query = "Find me trains from Delhi to Patna on next Tuesday"
    initial_state = {
        "origin": "Delhi",
        "destination": "Patna", 
        "departure_date": "2025-09-30",
        "mode": "train",
        "messages": [{"role": "user", "content": user_query}],
        "user_query": user_query,
        "train_results": [],
        "next_agent": "train_agent_node"
    }

    print("Running train search agent...")
    print(f"Query: {user_query}")
    print(f"Origin: {initial_state['origin']}")
    print(f"Destination: {initial_state['destination']}")
    print(f"Date: {initial_state['departure_date']}")
    
    final_state = train_chatbot.invoke(initial_state)
    
    print(f"\nNext Agent: {final_state.get('next_agent', 'Not set')}")
    print("\nFinal Response:")
    if final_state.get("messages"):
        last_message = final_state["messages"][-1]
        if hasattr(last_message, 'content'):
            print(last_message.content)
        else:
            print(str(last_message))   







