import os
import re
import dateparser
import requests
from datetime import datetime
from langchain.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from typing import TypedDict, List, Dict, Optional, Annotated
from enum import Enum
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langgraph.graph import MessagesState

# Load environment variables for API keys
load_dotenv()

CITY_TO_CODE = {
    "delhi": "NDLS",
    "new delhi": ["NDLS", "ANVT", "DLI", "NZM", "DEE", "DSA"],
    "patna": "PNBE",
    "mumbai": "CSTM",
    "kolkata": "HWH",
    "chennai": "MAS",
    # Add more as needed
}

class TransportMode(Enum):
    FLIGHT = "flight"
    BUS = "bus"
    TRAIN = "train"

class State(MessageState):
    next_agent: Optional[str]
    origin: Optional[str]
    destination: Optional[str]
    departure_date: Optional[str]
    return_date: Optional[str]
    departure_time: Optional[str]
    return_time: Optional[str]
    mode: Optional[TransportMode]
    train_results: Optional[List[Dict]]
    bus_results: Optional[List[Dict]]
    flight_results: Optional[List[Dict]]
    booking_options: Optional[list]
    selected_option: Optional[dict]
    booking_confirmed: Optional[bool]

def fetch_trains_by_day(date_str: str, source: str, destination: str) -> str:
    try:
        dt = dateparser.parse(date_str)
        if not dt:
            return f"Could not parse date: {date_str}"
        weekday_key = dt.strftime("%a").lower()[:3]
        headers = {
            'x-rapidapi-key': os.getenv("bd2eb2510fmshac0a7df8a5da592p128912jsn1d08c95ac607"),
            'x-rapidapi-host': "irctc1.p.rapidapi.com"
        }
        url = f"https://irctc1.p.rapidapi.com/api/v3/getLiveStation?fromStationCode={source}&toStationCode={destination}&hours=24"
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        trains = response.json().get("data", [])
        trains_today = [
            t for t in trains if t.get("runDays", {}).get(weekday_key, False)
        ]
        if not trains_today:
            return f"No trains found on {dt.strftime('%d-%m-%Y')} from {source} to {destination}."
        formatted = "\n".join(
            f"{t['trainName']} ({t['trainNumber']}) at {t['departureTime']}" for t in trains_today
        )
        return f"Available trains on {dt.strftime('%d-%m-%Y')} from {source} to {destination}:\n{formatted}"
    except Exception as e:
        return f"API error: {str(e)}"

@tool("train_options_tool")
def train_options_tool(date_str: str, source: str, destination: str) -> str:
    """
    Fetches available trains between two stations on a given date using the IRCTC RapidAPI.
    """
    return fetch_trains_by_day(date_str, source, destination)

tools = [train_options_tool]
graph_builder = StateGraph(State)
llm = init_chat_model("google_genai:gemini-2.0-flash")

def system_context(state: State) -> str:
    """Generate system context string from current state fields."""
    items = []
    if state.get("origin"):
        items.append(f"Origin: {state['origin']}")
    if state.get("destination"):
        items.append(f"Destination: {state['destination']}")
    if state.get("departure_date"):
        items.append(f"Departure Date: {state['departure_date']}")
    if state.get("mode"):
        items.append(f"Mode: {state['mode'].value}")
    return " | ".join(items)

def chatbot(state: State):
    # Inject summary context in every LLM call
    system_msg = {"role": "system", "content": system_context(state)}
    full_history = [system_msg] + state["messages"]
    return {"messages": state["messages"] + [llm.invoke(full_history)]}

tool_node = ToolNode(tools=tools)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile()

def resolve_city_code(city):
    code_data = CITY_TO_CODE.get(city.strip().lower())
    if not code_data:
        return city.strip().upper()
    if isinstance(code_data, list):
        return code_data[0]
    return code_data

def run_chatbot():
    state: State = {
        "messages": [],
        "next_agent": None,
        "origin": None,
        "destination": None,
        "departure_date": None,
        "return_date": None,
        "departure_time": None,
        "return_time": None,
        "mode": None,
        "train_results": None,
        "bus_results": None,
        "flight_results": None,
        "booking_options": None,
        "selected_option": None,
        "booking_confirmed": None,
    }
    print("Train Assistant: Type 'exit' to quit.")
    print("Ask: 'Give me some train options between delhi and patna on 21-09-2025' or '21-09-2025 NDLS PNBE'")
    exit_commands = {"exit", "bye", "goodbye", "quit", "close", "end"}
    while True:
        user_input = input("Message: ").strip()
        if user_input.lower() in exit_commands:
            print("Goodbye!")
            break

        # Date extraction (supports some natural dates)
        date_match = re.search(r'(\d{2}-\d{2}-\d{4}|\d{4}-\d{2}-\d{2}|tomorrow|today|next [a-zA-Z]+)', user_input, re.IGNORECASE)
        date_str = date_match.group(1) if date_match else ""

        # Origin/destination extraction
        city_matches = re.findall(r'between ([a-zA-Z ]+) and ([a-zA-Z ]+)', user_input, re.IGNORECASE)
        if city_matches:
            city1, city2 = city_matches[0]
            source = resolve_city_code(city1)
            destination = resolve_city_code(city2)
        else:
            code_matches = re.findall(r'([A-Z]{2,5})', user_input)
            if len(code_matches) >= 2:
                source, destination = code_matches[0], code_matches[1]
            else:
                source, destination = "", ""

        # Mode inference (only 'train' demo for now)
        state['mode'] = TransportMode.TRAIN if "train" in user_input.lower() else None
        state['origin'] = source if source else None
        state['destination'] = destination if destination else None
        state['departure_date'] = date_str if date_str else None

        # If all info for querying trains is present, call API directly
        if date_str and source and destination:
            tool_response = fetch_trains_by_day(date_str, source, destination)
            print(tool_response)
            state["messages"] += [{"role": "user", "content": user_input}, {"role": "assistant", "content": tool_response}]
            state["train_results"] = tool_response
            continue

        # Default: Let LLM resolve next step with injected state context
        state["messages"] += [{"role": "user", "content": user_input}]
        state = graph.invoke(state)
        if state.get("messages") and len(state["messages"]) > 0:
            last_message = state["messages"][-1]
            # Extract only the plain string reply, even if nested or wrapped
            def extract_content(msg):
                if isinstance(msg, dict):
                    # Try 'content' key first
                    if 'content' in msg and isinstance(msg['content'], str):
                        return msg['content']
                    # If 'content' is a dict, recurse
                    if 'content' in msg and isinstance(msg['content'], dict):
                        return extract_content(msg['content'])
                    # Sometimes reply is under 'text' or other keys
                    for key in ['text', 'reply', 'message']:
                        if key in msg and isinstance(msg[key], str):
                            return msg[key]
                        if key in msg and isinstance(msg[key], dict):
                            return extract_content(msg[key])
                    # Fallback: stringified dict
                    return str(msg)
                # If the string looks like 'content=...', extract after 'content='
                s = str(msg)
                if s.startswith('content='):
                    # Try to extract the quoted string after content=
                    match = re.search(r"content='([^']*)'", s)
                    if match:
                        return match.group(1)
                return s
            # Print only the reply text, with no extra labels or wrappers
            reply = extract_content(last_message)
            # Remove leading 'content="' or "content='" and trailing quote if present
            match = re.match(r"content=['\"]?(.*)['\"]?$", reply)
            if match:
                reply = match.group(1)
            print(reply)

if __name__ == "__main__":
    run_chatbot()



