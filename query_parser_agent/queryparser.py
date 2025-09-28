import os
import json
import datetime
import time
from typing import Any, Dict, TypedDict, Optional, List, Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage, SystemMessage
from langgraph.graph import StateGraph, END
from graph.state import State

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
2. Only update fields that have new information from the actual user input
3. Keep existing information if not mentioned in the query
4. Use standard formats: dates as YYYY-MM-DD, times as HH:MM
5. NEVER use placeholder values like [city name] or [YYYY-MM-DD]
6. If the user query contains actual information, extract it
7. If the user query is asking for information or is unclear, respond with NO_CHANGES

RESPONSE FORMAT:
Return ONLY the fields that need updating in this exact format:
ORIGIN: [actual city name from user input]
DESTINATION: [actual city name from user input]
DEPARTURE_DATE: [actual date in YYYY-MM-DD format]
RETURN_DATE: [actual date in YYYY-MM-DD format]
DEPARTURE_TIME: [actual time in HH:MM format]
RETURN_TIME: [actual time in HH:MM format]
MODE: [flight/bus/train]

If no new concrete information is found in the user's input, respond with: NO_CHANGES

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

User: "I need more information. Please provide your departure city, departure date"
Response:
NO_CHANGES
"""),
        ("human", "{query}")
    ])
    
    return parser_prompt | llm

def query_parser(state: State) -> Dict[str, Any]:
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
            response_msg = "I didn't receive a query. Please tell me about your travel plans."
            return {
                **state,  # Preserve all existing state
                "messages": messages + [AIMessage(content=response_msg)],
                "next_agent": "query_parser",
                "needs_user_input": True  # Add flag to indicate we need user input
            }
        
        # Check if this is a system-generated message asking for more info
        # If so, we should wait for user input rather than processing it
        if "I need more information" in query or "Please provide" in query:
            return {
                **state,
                "needs_user_input": True,
                "next_agent": "wait_for_input"  # Use a special state that stops processing
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
                    
                    # Remove any placeholder brackets like [city name] or template text
                    if (field_value.startswith('[') and field_value.endswith(']')) or \
                       field_value in ['[city name]', '[YYYY-MM-DD]', '[HH:MM]', '[actual city name from user input]', 
                                     '[actual date in YYYY-MM-DD format]', '[actual time in HH:MM format]']:
                        
                        continue
                    
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
        
        
        # Validate required fields - only check non-empty values
        required_fields = ["origin", "destination", "departure_date", "mode"]
        missing_fields = []
        
        for field in required_fields:
            value = final_state.get(field, "").strip()
            if not value or value.startswith('['):  # Also check for placeholder values
                missing_fields.append(field)
        
        # Determine response and next agent
        if missing_fields:
            field_prompts = {
                "origin": "departure city",
                "destination": "arrival city", 
                "departure_date": "departure date (YYYY-MM-DD format)",
                "mode": "travel mode (flight, bus, or train)"
            }
            
            missing_prompts: List[str] = [field_prompts.get(field, field) for field in missing_fields]
            missing_list: str = ", ".join(missing_prompts)
            
            response_msg = f"I need more information. Please provide your {missing_list}."
            next_agent = "wait_for_input"  # Stop processing and wait for user input
            needs_input = True
            
        else:
            # All required info collected
            summary = f"""Great! Here's your travel information:
✈️ From: {final_state['origin']} → {final_state['destination']}
📅 Date: {final_state['departure_date']}
🚗 Mode: {final_state['mode'].title()}"""
            
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
            
            needs_input = False
        
        # Add response message to state
        new_messages = messages + [AIMessage(content=response_msg)]
        
        # Return complete updated state
        result = {
            **state,  # Preserve any other state fields
            "messages": new_messages,
            "next_agent": next_agent,
            "needs_user_input": needs_input,
            **final_state  # Include all travel information (current + updated)
        }
        return result
        
    except Exception as e:
        # Ensure messages is defined for error case
        messages = state.get("messages", []) or []
        error_msg = f"Sorry, I encountered an error processing your request: {str(e)}"
        print(f"DEBUG: Error in query_parser: {str(e)}")
        return {
            **state,  # Preserve existing state
            "messages": messages + [AIMessage(content=error_msg)],
            "next_agent": "query_parser",
            "needs_user_input": True
        }

