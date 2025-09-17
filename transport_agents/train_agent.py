# Add these imports for parsing
import re
import dateparser

# Simple city-to-station-code mapping (expand as needed)
CITY_TO_CODE = {
    "delhi": "NDLS",
    "new delhi": ["NDLS", "ANVT", "DLI", "NZM", "DEE", "DSA"],
    "patna": "PNBE",
    "mumbai": "CSTM",
    "kolkata": "HWH",
    "chennai": "MAS",
    # Add more as needed
}
import requests
from datetime import datetime
from langchain.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import add_messages
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
import dotenv
from dotenv import load_dotenv
load_dotenv()


def fetch_trains_by_day(date_str: str, source: str, destination: str) -> str:
    try:
   
        dt = dateparser.parse(date_str)
        if not dt:
            return f"Could not parse date: {date_str}"
        weekday_key = dt.strftime("%a").lower()[:3] 
        headers = {
            'x-rapidapi-key': "bd2eb2510fmshac0a7df8a5da592p128912jsn1d08c95ac607",
            'x-rapidapi-host': "irctc1.p.rapidapi.com"
        }
        url = f"https://irctc1.p.rapidapi.com/api/v3/getLiveStation?fromStationCode={source}&toStationCode={destination}&hours=24"
        response = requests.get(url, headers=headers)
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
    Args:
        date_str: Date in DD-MM-YYYY, YYYY-MM-DD, or natural language (e.g., 'tomorrow').
        source: Source station code (e.g., 'NDLS').
        destination: Destination station code (e.g., 'PNBE').
    Returns:
        A string listing available trains or an error message.
    """
    return fetch_trains_by_day(date_str, source, destination)

tools = [train_options_tool]


class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)
llm = init_chat_model("google_genai:gemini-2.0-flash")

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

tool_node = ToolNode(tools=tools)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile()


def run_chatbot():
    state = {"messages": [], "message_type": None}
    print("Train Assistant: Type 'exit' to quit.")
    print("You can ask: 'Give me some train options between delhi and patna on 21-09-2025' or '21-09-2025 NDLS PNBE'")
    exit_commands = ["exit", "bye", "goodbye", "quit", "close", "end"]
    while True:
        user_input = input("Message: ")
        if user_input.strip().lower() in exit_commands:
            print("Goodbye!")
            break

       
        date_match = re.search(r'(\d{2}-\d{2}-\d{4}|\d{4}-\d{2}-\d{2}|tomorrow|today|next [a-zA-Z]+)', user_input, re.IGNORECASE)
        date_str = date_match.group(1) if date_match else ""


        city_matches = re.findall(r'between ([a-zA-Z ]+) and ([a-zA-Z ]+)', user_input, re.IGNORECASE)
        if city_matches:
            city1, city2 = city_matches[0]
            source = CITY_TO_CODE.get(city1.strip().lower(), city1.strip().upper())
            destination = CITY_TO_CODE.get(city2.strip().lower(), city2.strip().upper())
        else:
           
            code_matches = re.findall(r'([A-Z]{2,5})', user_input)
            if len(code_matches) >= 2:
                source, destination = code_matches[0], code_matches[1]
            else:
                source, destination = "", ""

        if date_str and source and destination:
            tool_response = fetch_trains_by_day(date_str, source, destination)
            print(f"Assistant: {tool_response}")
        
            state["messages"] = state.get("messages", []) + [
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": tool_response}
            ]
            continue

        state["messages"] = state.get("messages", []) + [
            {"role": "user", "content": user_input}
        ]
        state = graph.invoke(state)
        if state.get("messages") and len(state["messages"]) > 0:
            last_message = state["messages"][-1]
            print(f"Assistant: {last_message.content}")

if __name__ == "__main__":
    run_chatbot()
