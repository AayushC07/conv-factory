"""Shared test fixtures and sample ToolBench data."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

SAMPLE_TOOLBENCH_TOOLS = [
    {
        "tool_name": "Hotels_API",
        "tool_description": "Search and book hotels worldwide",
        "category_name": "Travel",
        "home_url": "https://hotels-api.example.com",
        "api_list": [
            {
                "name": "search_hotels",
                "api_description": "Search for available hotels in a city",
                "method": "GET",
                "required_parameters": [
                    {"name": "city", "type": "string", "description": "City name"},
                ],
                "optional_parameters": [
                    {"name": "max_price", "type": "number", "description": "Maximum price per night"},
                    {"name": "currency", "type": "string", "description": "Currency code", "default": "USD"},
                    {"name": "check_in", "type": "string", "description": "Check-in date (YYYY-MM-DD)"},
                    {"name": "check_out", "type": "string", "description": "Check-out date (YYYY-MM-DD)"},
                    {"name": "guests", "type": "integer", "description": "Number of guests", "default": 1},
                ],
            },
            {
                "name": "get_hotel_details",
                "api_description": "Get detailed information about a specific hotel",
                "method": "GET",
                "required_parameters": [
                    {"name": "hotel_id", "type": "string", "description": "Hotel identifier"},
                ],
                "optional_parameters": [],
            },
            {
                "name": "book_hotel",
                "api_description": "Book a hotel room",
                "method": "POST",
                "required_parameters": [
                    {"name": "hotel_id", "type": "string", "description": "Hotel identifier"},
                    {"name": "check_in", "type": "string", "description": "Check-in date"},
                    {"name": "check_out", "type": "string", "description": "Check-out date"},
                    {"name": "guest_name", "type": "string", "description": "Guest full name"},
                ],
                "optional_parameters": [
                    {"name": "special_requests", "type": "string", "description": "Special requests"},
                ],
            },
        ],
    },
    {
        "tool_name": "Flights_API",
        "tool_description": "Search and book flights",
        "category_name": "Travel",
        "api_list": [
            {
                "name": "search_flights",
                "api_description": "Search for available flights",
                "method": "GET",
                "required_parameters": [
                    {"name": "origin", "type": "string", "description": "Origin airport code"},
                    {"name": "destination", "type": "string", "description": "Destination airport code"},
                    {"name": "date", "type": "string", "description": "Travel date"},
                ],
                "optional_parameters": [
                    {"name": "max_price", "type": "number", "description": "Maximum price"},
                    {"name": "class", "type": "string", "description": "Travel class"},
                ],
            },
            {
                "name": "book_flight",
                "api_description": "Book a flight",
                "method": "POST",
                "required_parameters": [
                    {"name": "flight_id", "type": "string", "description": "Flight identifier"},
                    {"name": "passenger_name", "type": "string", "description": "Passenger name"},
                ],
                "optional_parameters": [
                    {"name": "seat_preference", "type": "string", "description": "Seat preference"},
                ],
            },
        ],
    },
    {
        "tool_name": "Weather_API",
        "tool_description": "Get weather information for any location",
        "category_name": "Weather",
        "api_list": [
            {
                "name": "get_current_weather",
                "api_description": "Get current weather for a location",
                "method": "GET",
                "required_parameters": [
                    {"name": "city", "type": "string", "description": "City name"},
                ],
                "optional_parameters": [
                    {"name": "units", "type": "string", "description": "Temperature units (celsius/fahrenheit)"},
                ],
            },
            {
                "name": "get_forecast",
                "api_description": "Get weather forecast for upcoming days",
                "method": "GET",
                "required_parameters": [
                    {"name": "city", "type": "string", "description": "City name"},
                ],
                "optional_parameters": [
                    {"name": "days", "type": "integer", "description": "Number of days", "default": 5},
                ],
            },
        ],
    },
    {
        "tool_name": "Restaurant_API",
        "tool_description": "Find and reserve restaurants",
        "category_name": "Food",
        "api_list": [
            {
                "name": "search_restaurants",
                "api_description": "Search for restaurants in an area",
                "method": "GET",
                "required_parameters": [
                    {"name": "location", "type": "string", "description": "Location or city"},
                ],
                "optional_parameters": [
                    {"name": "cuisine", "type": "string", "description": "Type of cuisine"},
                    {"name": "price_range", "type": "string", "description": "Price range (low/medium/high)"},
                    {"name": "rating_min", "type": "number", "description": "Minimum rating"},
                ],
            },
            {
                "name": "get_restaurant_details",
                "api_description": "Get details about a restaurant",
                "method": "GET",
                "required_parameters": [
                    {"name": "restaurant_id", "type": "string", "description": "Restaurant identifier"},
                ],
                "optional_parameters": [],
            },
            {
                "name": "make_reservation",
                "api_description": "Make a restaurant reservation",
                "method": "POST",
                "required_parameters": [
                    {"name": "restaurant_id", "type": "string", "description": "Restaurant identifier"},
                    {"name": "date", "type": "string", "description": "Reservation date"},
                    {"name": "time", "type": "string", "description": "Reservation time"},
                    {"name": "party_size", "type": "integer", "description": "Number of guests"},
                ],
                "optional_parameters": [
                    {"name": "name", "type": "string", "description": "Reservation name"},
                ],
            },
        ],
    },
    {
        "tool_name": "Maps_API",
        "tool_description": "Get directions and location information",
        "category_name": "Navigation",
        "api_list": [
            {
                "name": "get_directions",
                "api_description": "Get directions between two locations",
                "method": "GET",
                "required_parameters": [
                    {"name": "origin", "type": "string", "description": "Starting location"},
                    {"name": "destination", "type": "string", "description": "End location"},
                ],
                "optional_parameters": [
                    {"name": "mode", "type": "string", "description": "Travel mode (driving/walking/transit)"},
                ],
            },
            {
                "name": "search_places",
                "api_description": "Search for places near a location",
                "method": "GET",
                "required_parameters": [
                    {"name": "query", "type": "string", "description": "Search query"},
                    {"name": "location", "type": "string", "description": "Location to search near"},
                ],
                "optional_parameters": [
                    {"name": "radius", "type": "integer", "description": "Search radius in meters"},
                ],
            },
        ],
    },
    {
        "tool_name": "Events_API",
        "tool_description": "Find events and buy tickets",
        "category_name": "Entertainment",
        "api_list": [
            {
                "name": "search_events",
                "api_description": "Search for events in a city",
                "method": "GET",
                "required_parameters": [
                    {"name": "city", "type": "string", "description": "City name"},
                ],
                "optional_parameters": [
                    {"name": "category", "type": "string", "description": "Event category"},
                    {"name": "date_from", "type": "string", "description": "Start date"},
                    {"name": "date_to", "type": "string", "description": "End date"},
                ],
            },
            {
                "name": "get_event_details",
                "api_description": "Get event details",
                "method": "GET",
                "required_parameters": [
                    {"name": "event_id", "type": "string", "description": "Event identifier"},
                ],
                "optional_parameters": [],
            },
            {
                "name": "purchase_tickets",
                "api_description": "Purchase tickets for an event",
                "method": "POST",
                "required_parameters": [
                    {"name": "event_id", "type": "string", "description": "Event identifier"},
                    {"name": "quantity", "type": "integer", "description": "Number of tickets"},
                ],
                "optional_parameters": [
                    {"name": "ticket_type", "type": "string", "description": "Ticket type (general/vip)"},
                ],
            },
        ],
    },
]


def create_sample_toolbench_dir() -> str:
    """Create a temp directory with sample ToolBench-format data."""
    tmpdir = tempfile.mkdtemp(prefix="toolbench_test_")
    data_file = Path(tmpdir) / "tools.json"
    with open(data_file, "w") as f:
        json.dump(SAMPLE_TOOLBENCH_TOOLS, f, indent=2)
    return tmpdir


def create_category_layout_dir() -> str:
    """Create a temp directory with category-directory layout."""
    tmpdir = tempfile.mkdtemp(prefix="toolbench_cat_test_")
    for tool in SAMPLE_TOOLBENCH_TOOLS:
        cat = tool.get("category_name", "Other")
        cat_dir = Path(tmpdir) / cat
        cat_dir.mkdir(exist_ok=True)
        tool_file = cat_dir / f"{tool['tool_name']}.json"
        with open(tool_file, "w") as f:
            json.dump(tool, f, indent=2)
    return tmpdir


def build_test_registry():
    """Build a registry from sample data for testing."""
    from conversation_factory.registry import ToolBenchLoader, ToolRegistry

    tmpdir = create_sample_toolbench_dir()
    loader = ToolBenchLoader(tmpdir)
    tools = loader.load_all()
    registry = ToolRegistry()
    registry.add_tools(tools)
    return registry


def build_test_graph(registry=None):
    """Build a graph from sample data for testing."""
    from conversation_factory.graph import ToolGraph

    if registry is None:
        registry = build_test_registry()
    graph = ToolGraph()
    graph.build(registry)
    return graph, registry
