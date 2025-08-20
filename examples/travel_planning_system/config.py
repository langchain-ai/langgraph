"""
Configuration management for the Travel Planning System.
Handles API keys, endpoints, and system settings.
"""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class APIConfig:
    """Configuration for external APIs."""
    
    # OpenWeatherMap API (free tier: 1000 calls/day)
    openweather_api_key: Optional[str] = None
    openweather_base_url: str = "https://api.openweathermap.org/data/2.5"
    
    # Amadeus API (free tier: 2000 calls/month) - for flights
    amadeus_api_key: Optional[str] = None
    amadeus_api_secret: Optional[str] = None
    amadeus_base_url: str = "https://test.api.amadeus.com"
    
    # ExchangeRate API (free tier: 1500 calls/month)
    exchangerate_api_key: Optional[str] = None
    exchangerate_base_url: str = "https://v6.exchangerate-api.com/v6"
    
    # Foursquare Places API (free tier: 100 calls/day)
    foursquare_api_key: Optional[str] = None
    foursquare_base_url: str = "https://api.foursquare.com/v3/places"
    
    # Eventbrite API (free)
    eventbrite_token: Optional[str] = None
    eventbrite_base_url: str = "https://www.eventbriteapi.com/v3"
    
    def __post_init__(self):
        """Load API keys from environment variables."""
        self.openweather_api_key = os.getenv("OPENWEATHER_API_KEY")
        self.amadeus_api_key = os.getenv("AMADEUS_API_KEY")
        self.amadeus_api_secret = os.getenv("AMADEUS_API_SECRET")
        self.exchangerate_api_key = os.getenv("EXCHANGERATE_API_KEY")
        self.foursquare_api_key = os.getenv("FOURSQUARE_API_KEY")
        self.eventbrite_token = os.getenv("EVENTBRITE_TOKEN")

@dataclass
class LLMConfig:
    """Configuration for LLM providers."""
    
    # Azure OpenAI
    azure_openai_api_key: Optional[str] = None
    azure_openai_endpoint: Optional[str] = None
    azure_openai_api_version: str = "2024-02-15-preview"
    azure_deployment_name: str = "gpt-4"
    
    # OpenAI (fallback)
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4"
    
    # Temperature settings for different agents
    supervisor_temperature: float = 0.3
    flight_specialist_temperature: float = 0.5
    hotel_specialist_temperature: float = 0.5
    activity_specialist_temperature: float = 0.7
    budget_analyst_temperature: float = 0.2
    
    def __post_init__(self):
        """Load LLM config from environment variables."""
        self.azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION", self.azure_openai_api_version)
        self.azure_deployment_name = os.getenv("AZURE_DEPLOYMENT_NAME", self.azure_deployment_name)
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

@dataclass
class TracingConfig:
    """Configuration for tracing and observability."""
    
    # Azure Application Insights
    application_insights_connection_string: Optional[str] = None
    
    # OpenTelemetry
    otel_exporter_otlp_endpoint: str = "http://localhost:4318/v1/traces"
    otel_service_name: str = "multi-agent-travel-planning"
    
    # Enable content recording (be careful with PII)
    enable_content_recording: bool = True
    
    def __post_init__(self):
        """Load tracing config from environment variables."""
        self.application_insights_connection_string = os.getenv("APPLICATION_INSIGHTS_CONNECTION_STRING")
        self.otel_exporter_otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", self.otel_exporter_otlp_endpoint)
        self.otel_service_name = os.getenv("OTEL_SERVICE_NAME", self.otel_service_name)
        enable_recording = os.getenv("AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED", "true")
        self.enable_content_recording = enable_recording.lower() == "true"

@dataclass
class TravelConfig:
    """Configuration for travel planning defaults."""
    
    # Default values for planning
    default_trip_duration_days: int = 7
    default_travelers: int = 2
    default_budget_per_person: float = 2000.0
    
    # Fallback cities and airports
    major_airports: Dict[str, str] = None
    popular_destinations: Dict[str, Dict] = None
    
    def __post_init__(self):
        """Initialize travel data."""
        self.major_airports = {
            "Seattle": "SEA",
            "New York": "JFK", 
            "Los Angeles": "LAX",
            "Chicago": "ORD",
            "San Francisco": "SFO",
            "Miami": "MIA",
            "London": "LHR",
            "Paris": "CDG",
            "Tokyo": "NRT",
            "Rome": "FCO",
            "Sydney": "SYD",
            "Dubai": "DXB"
        }
        
        self.popular_destinations = {
            "London": {
                "country": "United Kingdom",
                "currency": "GBP",
                "timezone": "Europe/London",
                "coordinates": {"lat": 51.5074, "lon": -0.1278}
            },
            "Paris": {
                "country": "France", 
                "currency": "EUR",
                "timezone": "Europe/Paris",
                "coordinates": {"lat": 48.8566, "lon": 2.3522}
            },
            "New York": {
                "country": "United States",
                "currency": "USD", 
                "timezone": "America/New_York",
                "coordinates": {"lat": 40.7128, "lon": -74.0060}
            },
            "Tokyo": {
                "country": "Japan",
                "currency": "JPY",
                "timezone": "Asia/Tokyo", 
                "coordinates": {"lat": 35.6762, "lon": 139.6503}
            },
            "Rome": {
                "country": "Italy",
                "currency": "EUR",
                "timezone": "Europe/Rome",
                "coordinates": {"lat": 41.9028, "lon": 12.4964}
            }
        }

class ConfigManager:
    """Centralized configuration manager."""
    
    def __init__(self):
        self.api = APIConfig()
        self.llm = LLMConfig()
        self.tracing = TracingConfig()
        self.travel = TravelConfig()
    
    def validate_required_config(self) -> Dict[str, bool]:
        """Validate that required configuration is available."""
        validation_results = {
            "llm_configured": self.llm.azure_openai_api_key is not None or self.llm.openai_api_key is not None,
            "weather_api": self.api.openweather_api_key is not None,
            "flight_api": self.api.amadeus_api_key is not None and self.api.amadeus_api_secret is not None,
            "currency_api": self.api.exchangerate_api_key is not None,
            "places_api": self.api.foursquare_api_key is not None,
            "events_api": self.api.eventbrite_token is not None,
            "tracing_configured": (
                self.tracing.application_insights_connection_string is not None or 
                self.tracing.otel_exporter_otlp_endpoint is not None
            )
        }
        return validation_results
    
    def get_missing_config_instructions(self) -> str:
        """Get instructions for missing configuration."""
        validation = self.validate_required_config()
        missing = [key for key, value in validation.items() if not value]
        
        if not missing:
            return "All configuration is properly set up!"
        
        instructions = ["Missing configuration detected. Please set the following environment variables:\n"]
        
        if not validation["llm_configured"]:
            instructions.append("LLM Configuration (choose one):")
            instructions.append("  - AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT (for Azure OpenAI)")
            instructions.append("  - OPENAI_API_KEY (for OpenAI)")
            instructions.append("")
        
        if not validation["weather_api"]:
            instructions.append("Weather API:")
            instructions.append("  - OPENWEATHER_API_KEY (get free at https://openweathermap.org/api)")
            instructions.append("")
        
        if not validation["flight_api"]:
            instructions.append("Flight API:")
            instructions.append("  - AMADEUS_API_KEY and AMADEUS_API_SECRET (get free at https://developers.amadeus.com)")
            instructions.append("")
        
        if not validation["currency_api"]:
            instructions.append("Currency API:")
            instructions.append("  - EXCHANGERATE_API_KEY (get free at https://exchangerate-api.com)")
            instructions.append("")
        
        if not validation["places_api"]:
            instructions.append("Places API:")
            instructions.append("  - FOURSQUARE_API_KEY (get free at https://developer.foursquare.com)")
            instructions.append("")
        
        if not validation["events_api"]:
            instructions.append("Events API:")
            instructions.append("  - EVENTBRITE_TOKEN (get free at https://www.eventbrite.com/platform/api)")
            instructions.append("")
        
        instructions.append("Optional - Tracing:")
        instructions.append("  - APPLICATION_INSIGHTS_CONNECTION_STRING (for Azure Application Insights)")
        instructions.append("  - OTEL_EXPORTER_OTLP_ENDPOINT (for OpenTelemetry, default: http://localhost:4318/v1/traces)")
        
        return "\n".join(instructions)

# Global configuration instance
config = ConfigManager()