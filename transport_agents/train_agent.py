import os
import requests
import dateparser
from typing import Any, Dict, List, Optional

# LangChain & LangGraph Imports
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool

# Assuming 'State' is defined in a shared file like 'graph/state.py'
# from graph.state import State 
# Using a placeholder for standalone execution:
class State(dict):
    """A dictionary-based placeholder for the application's state."""
    pass

# ---
# 1. SETUP: Initialize LLM 
# ---

# Ensure your GOOGLE_API_KEY is set in your environment
try:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
except Exception as e:
    print(f"Error initializing LLM. Make sure GOOGLE_API_KEY is set. Details: {e}")
    llm = None

# ---
# 2. TOOL DEFINITION: The tool for fetching train data
# ---

CITY_TO_CODE = {
    "delhi": "NDLS",
    "new delhi": "NDLS",
    "patna": "PNBE",
    "mumbai": "CSTM",
    "kolkata": "HWH",
    "chennai": "MAS",
}

def fetch_trains_by_day(date_str: str, source: str, destination: str) -> str:
    """Helper function to call the train API."""
    try:
        dt = dateparser.parse(date_str)
        if not dt:
            return f"Could not parse date: {date_str}"
        weekday_key = dt.strftime("%a").lower()[:3]

        api_key = os.getenv("RAPIDAPI_KEY")
        if not api_key:
            return "API key 'RAPIDAPI_KEY' not found. Please set it in your environment."

        headers = {
            'x-rapidapi-key': api_key,
            'x-rapidapi-host': "irctc1.p.rapidapi.com"
        }
        url = f"https://irctc1.p.rapidapi.com/api/v3/getLiveStation?fromStationCode={source}&toStationCode={destination}&hours=8"
        
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()

        trains = data.get("data", [])
        if not trains:
            return f"No trains were found in the API response for the route from {source} to {destination}."

        trains_today = [t for t in trains if t.get("runDays", {}).get(weekday_key, False)]
        
        if not trains_today:
            return f"No trains are scheduled to run on {dt.strftime('%A, %d-%m-%Y')} from {source} to {destination}."

        formatted = "\n".join(f"- {t['trainName']} ({t['trainNumber']}) departing at {t['departureTime']}" for t in trains_today)
        return f"Available trains on {dt.strftime('%d-%m-%Y')} from {source} to {destination}:\n{formatted}"
    except requests.exceptions.HTTPError as http_err:
        return f"An HTTP error occurred: {http_err}. The API might be unavailable or the station codes might be invalid."
    except Exception as e:
        return f"An unexpected error occurred while fetching train data: {str(e)}"

@tool
def train_options_tool(date_str: str, source: str, destination: str) -> str:
    """
    Fetches available trains between two stations on a given date.
    
    Args:
        date_str: The date of travel in a recognizable format (e.g., '2024-12-25', 'tomorrow').
        source: The source city name (e.g., 'Delhi', 'Patna').
        destination: The destination city name (e.g., 'Mumbai', 'Kolkata').
    """
    def resolve_city_code(city):
        """Resolves city name to station code."""
        code = CITY_TO_CODE.get(city.strip().lower(), city.strip().upper())
        # Handles cases where a city has multiple codes, just takes the first
        return code[0] if isinstance(code, list) else code

    source_code = resolve_city_code(source)
    destination_code = resolve_city_code(destination)
    
    return fetch_trains_by_day(date_str, source_code, destination_code)


# ---
# 3. TRAIN AGENT NODE
# ---

def train_agent(state: State) -> Dict[str, Any]:
    """
    A specialist agent for handling train-related queries.
    
    This agent assumes the state has been populated with origin, destination,
    and departure_date by a previous node (like a query parser).
    
    It operates in two stages:
    1.  If the last message is not a ToolMessage, it calls the `train_options_tool`.
    2.  If the last message is a ToolMessage, it summarizes the results for the user.
    """
    messages = state["messages"]
    
    # Stage 2: Summarize the results from the tool call.
    if isinstance(messages[-1], ToolMessage):
        # Create a prompt to summarize the tool's output.
        summary_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful travel assistant. Summarize the provided train options for the user in a clear and friendly manner. If no trains were found, state that and apologize for the inconvenience."),
            ("user", "Here are the train search results:\n\n{tool_output}")
        ])
        
        # Chain the prompt with the LLM to generate the summary.
        chain = summary_prompt | llm
        response = chain.invoke({"tool_output": messages[-1].content})
        
        # Append the summary to the messages and prepare to end this branch.
        return {"messages": messages + [AIMessage(content=response.content)]}

    # Stage 1: Call the tool with information from the state.
    # Bind the tool to the LLM so it knows how to call it.
    llm_with_tools = llm.bind_tools([train_options_tool])
    
    # Create a prompt that instructs the LLM to use the tool.
    # We pass the state variables directly to the tool call, making it deterministic.
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a travel assistant whose only job is to find train options. Use the `train_options_tool` to search for trains based on the user's confirmed travel plans."),
        # This user message is a template that will be filled from the state.
        ("user", "Please find trains from {origin} to {destination} on {departure_date}.")
    ])

    # Create the chain to invoke the tool.
    chain = prompt | llm_with_tools
    
    # Invoke the chain with the necessary info from the state.
    # This will generate an AIMessage containing the tool call.
    ai_message_with_tool_call = chain.invoke({
        "origin": state['origin'],
        "destination": state['destination'],
        "departure_date": state['departure_date']
    })
    
    # Append the AI's tool-calling message to the history.
    # The graph will see this and route to the ToolNode next.
    return {"messages": messages + [ai_message_with_tool_call]}




