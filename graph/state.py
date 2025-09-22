from typing import TypedDict, List, Dict, Optional
from enum import Enum
from langgraph.graph import MessagesState

class TransportMode(Enum):
    FLIGHT = "flight"
    BUS = "bus"
    TRAIN = "train"

class State(MessagesState):
    # Input
    next_agent: str
    origin: Optional[str]
    destination: Optional[str]
    departure_date: Optional[str]
    return_date: Optional[str]
    departure_time: Optional[str]
    return_time: Optional[str]
    mode: Optional[TransportMode]

    # Output / processing
    train_results: Optional[List[Dict]]
    bus_results: Optional[List[Dict]]
    flight_results: Optional[List[Dict]]
    booking_options: list
    selected_option: dict
    booking_confirmed: bool
    needs_user_input: bool 