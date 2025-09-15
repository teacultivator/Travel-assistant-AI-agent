from typing import TypedDict, List, Dict, Any, Optional
from datetime import datetime

class State(TypedDict):
    # Input
    query: str
    parsed_query: Optional[Dict[str, Any]]  # Complete parsed structure from Gemini
    origin: Optional[str]                   # Departure location
    destination: Optional[str]              # Arrival location
    departure_date: Optional[str]           # YYYY-MM-DD format
    return_date: Optional[str]              # YYYY-MM-DD format or null
    departure_time: Optional[str]           # HH:MM format or null
    return_time: Optional[str]              # HH:MM format or null
    passengers: Optional[int]               # Number of passengers
    modes: Optional[List[str]]              # Transport modes: [train, bus, flight]
    trip_type: Optional[str]                # one-way or round-trip
    preferences: Optional[Dict[str, Any]]   # Budget, direct, fastest preferences
    reasoning: Optional[List[str]]          # AI reasoning for parsing decisions
    
    # Transport results (existing)
    train_results: Optional[List[Dict]]
    bus_results: Optional[List[Dict]]
    flight_results: Optional[List[Dict]]
    
    # Output (existing)
    formatted_output: Optional[str]
    analysis: Optional[str]
    
    # Error handling (enhanced)
    error: Optional[str]
    warnings: Optional[List[str]]           # Non-critical issues
    
    # Metadata (new)
    processing_time: Optional[float]        # Time taken for processing
    confidence_score: Optional[float]       # AI confidence in parsing (0-1)
    timestamp: Optional[datetime]           # When query was processed