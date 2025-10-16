"""
Real data sources for the Travel Planning System.
Integrates with free APIs and provides fallback data when APIs are unavailable.
"""

import json
import random
import asyncio
import aiohttp
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import time

from config import config

logger = logging.getLogger(__name__)

@dataclass
class FlightInfo:
    """Flight information structure."""
    flight_id: str
    airline: str
    departure_time: str
    arrival_time: str
    price: float
    duration: str
    origin: str
    destination: str
    date: str
    available_seats: int
    currency: str = "USD"

@dataclass
class HotelInfo:
    """Hotel information structure."""
    hotel_id: str
    name: str
    price_per_night: float
    rating: float
    amenities: List[str]
    location: str
    distance_to_center: Optional[str] = None
    currency: str = "USD"

@dataclass
class ActivityInfo:
    """Activity information structure."""
    activity_id: str
    name: str
    category: str
    price: float
    duration: str
    rating: float
    description: str
    currency: str = "USD"

@dataclass
class WeatherInfo:
    """Weather information structure."""
    location: str
    date: str
    condition: str
    temperature_high: int
    temperature_low: int
    humidity: int
    description: str

class DataSourceManager:
    """Manages all external data sources with fallbacks."""
    
    def __init__(self):
        self.amadeus_token = None
        self.amadeus_token_expires = None
        self.session = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def get_amadeus_token(self) -> Optional[str]:
        """Get Amadeus API access token."""
        if not config.api.amadeus_api_key or not config.api.amadeus_api_secret:
            return None
            
        # Check if existing token is still valid
        if (self.amadeus_token and self.amadeus_token_expires and 
            datetime.now() < self.amadeus_token_expires):
            return self.amadeus_token
        
        try:
            url = f"{config.api.amadeus_base_url}/v1/security/oauth2/token"
            data = {
                'grant_type': 'client_credentials',
                'client_id': config.api.amadeus_api_key,
                'client_secret': config.api.amadeus_api_secret
            }
            
            async with self.session.post(url, data=data) as response:
                if response.status == 200:
                    token_data = await response.json()
                    self.amadeus_token = token_data['access_token']
                    # Token expires in seconds, add buffer
                    expires_in = token_data.get('expires_in', 1800) - 60
                    self.amadeus_token_expires = datetime.now() + timedelta(seconds=expires_in)
                    return self.amadeus_token
                else:
                    logger.warning(f"Failed to get Amadeus token: {response.status}")
                    return None
        except Exception as e:
            logger.warning(f"Error getting Amadeus token: {e}")
            return None
    
    async def search_flights_api(self, origin: str, destination: str, departure_date: str, 
                                travelers: int = 1) -> List[FlightInfo]:
        """Search for flights using Amadeus API."""
        token = await self.get_amadeus_token()
        if not token:
            return []
        
        try:
            # Convert city names to airport codes
            origin_code = config.travel.major_airports.get(origin, origin)
            dest_code = config.travel.major_airports.get(destination, destination)
            
            url = f"{config.api.amadeus_base_url}/v2/shopping/flight-offers"
            params = {
                'originLocationCode': origin_code,
                'destinationLocationCode': dest_code,
                'departureDate': departure_date,
                'adults': travelers,
                'max': 5
            }
            
            headers = {'Authorization': f'Bearer {token}'}
            
            async with self.session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    flights = []
                    
                    for offer in data.get('data', []):
                        for itinerary in offer.get('itineraries', []):
                            first_segment = itinerary['segments'][0]
                            last_segment = itinerary['segments'][-1]
                            
                            flight = FlightInfo(
                                flight_id=offer['id'],
                                airline=first_segment['carrierCode'],
                                departure_time=first_segment['departure']['at'][-8:-3],  # Extract time
                                arrival_time=last_segment['arrival']['at'][-8:-3],
                                price=float(offer['price']['grandTotal']),
                                duration=itinerary['duration'][2:],  # Remove 'PT' prefix
                                origin=origin,
                                destination=destination,
                                date=departure_date,
                                available_seats=offer.get('numberOfBookableSeats', 9),
                                currency=offer['price']['currency']
                            )
                            flights.append(flight)
                    
                    return flights[:3]  # Return top 3 results
                else:
                    logger.warning(f"Amadeus API error: {response.status}")
                    return []
        except Exception as e:
            logger.warning(f"Error calling Amadeus API: {e}")
            return []
    
    def search_flights_fallback(self, origin: str, destination: str, departure_date: str, 
                               travelers: int = 1) -> List[FlightInfo]:
        """Fallback flight search with realistic data."""
        # Determine if international flight
        international_destinations = ['London', 'Paris', 'Tokyo', 'Rome', 'Sydney', 'Dubai']
        is_international = destination in international_destinations or origin in international_destinations
        
        airlines = {
            'international': ['British Airways', 'Virgin Atlantic', 'United', 'American', 'Delta', 'Lufthansa'],
            'domestic': ['United', 'Delta', 'American', 'Southwest', 'JetBlue', 'Alaska']
        }
        
        flight_airlines = airlines['international'] if is_international else airlines['domestic']
        base_price = random.randint(800, 2000) if is_international else random.randint(200, 800)
        
        flights = []
        for i in range(3):
            price_variation = random.uniform(0.8, 1.3)
            flight = FlightInfo(
                flight_id=f"FL{random.randint(1000, 9999)}",
                airline=random.choice(flight_airlines),
                departure_time=f"{random.randint(6, 22):02d}:{random.choice(['00', '30'])}",
                arrival_time=f"{random.randint(8, 23):02d}:{random.choice(['00', '30'])}",
                price=round(base_price * price_variation),
                duration=f"{random.randint(8, 15)}h {random.randint(0, 59)}m" if is_international else f"{random.randint(2, 8)}h {random.randint(0, 59)}m",
                origin=origin,
                destination=destination,
                date=departure_date,
                available_seats=random.randint(1, 15),
                currency="USD"
            )
            flights.append(flight)
        
        return flights
    
    async def search_flights(self, origin: str, destination: str, departure_date: str, 
                           travelers: int = 1) -> List[FlightInfo]:
        """Search for flights with API first, fallback to generated data."""
        # Try API first
        flights = await self.search_flights_api(origin, destination, departure_date, travelers)
        
        # Use fallback if API failed
        if not flights:
            logger.info("Using fallback flight data")
            flights = self.search_flights_fallback(origin, destination, departure_date, travelers)
        
        return flights
    
    async def get_weather_api(self, location: str, date: str) -> Optional[WeatherInfo]:
        """Get weather information using OpenWeatherMap API."""
        if not config.api.openweather_api_key:
            return None
        
        try:
            # Get coordinates for location
            dest_info = config.travel.popular_destinations.get(location)
            if not dest_info:
                return None
            
            lat, lon = dest_info["coordinates"]["lat"], dest_info["coordinates"]["lon"]
            
            url = f"{config.api.openweather_base_url}/weather"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': config.api.openweather_api_key,
                'units': 'metric'
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return WeatherInfo(
                        location=location,
                        date=date,
                        condition=data['weather'][0]['main'],
                        temperature_high=int(data['main']['temp_max']),
                        temperature_low=int(data['main']['temp_min']),
                        humidity=data['main']['humidity'],
                        description=data['weather'][0]['description']
                    )
                else:
                    logger.warning(f"OpenWeather API error: {response.status}")
                    return None
        except Exception as e:
            logger.warning(f"Error calling OpenWeather API: {e}")
            return None
    
    def get_weather_fallback(self, location: str, date: str) -> WeatherInfo:
        """Fallback weather with realistic seasonal data."""
        # Basic seasonal logic
        month = datetime.now().month
        
        if location in ['London', 'Paris']:
            if month in [12, 1, 2]:  # Winter
                temp_range = (2, 8)
                conditions = ['Cloudy', 'Light Rain', 'Overcast']
            elif month in [6, 7, 8]:  # Summer
                temp_range = (15, 25)
                conditions = ['Partly Cloudy', 'Sunny', 'Cloudy']
            else:
                temp_range = (8, 18)
                conditions = ['Partly Cloudy', 'Cloudy', 'Light Rain']
        else:
            # Default to moderate climate
            temp_range = (15, 28)
            conditions = ['Sunny', 'Partly Cloudy', 'Clear']
        
        temp_high = random.randint(*temp_range)
        temp_low = temp_high - random.randint(3, 8)
        condition = random.choice(conditions)
        
        return WeatherInfo(
            location=location,
            date=date,
            condition=condition,
            temperature_high=temp_high,
            temperature_low=temp_low,
            humidity=random.randint(40, 80),
            description=condition.lower()
        )
    
    async def get_weather(self, location: str, date: str) -> WeatherInfo:
        """Get weather with API first, fallback to generated data."""
        weather = await self.get_weather_api(location, date)
        
        if not weather:
            logger.info("Using fallback weather data")
            weather = self.get_weather_fallback(location, date)
        
        return weather
    
    async def search_hotels_api(self, destination: str, check_in: str, check_out: str, 
                              travelers: int = 1) -> List[HotelInfo]:
        """Search hotels using Foursquare Places API."""
        if not config.api.foursquare_api_key:
            return []
        
        try:
            # Get coordinates for destination
            dest_info = config.travel.popular_destinations.get(destination)
            if not dest_info:
                return []
            
            lat, lon = dest_info["coordinates"]["lat"], dest_info["coordinates"]["lon"]
            
            url = f"{config.api.foursquare_base_url}/search"
            params = {
                'll': f"{lat},{lon}",
                'categories': '19014',  # Hotels category
                'limit': 10
            }
            
            headers = {
                'Authorization': config.api.foursquare_api_key,
                'Accept': 'application/json'
            }
            
            async with self.session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    hotels = []
                    
                    for place in data.get('results', []):
                        # Calculate realistic pricing based on destination
                        base_price = self._get_hotel_base_price(destination)
                        price_variation = random.uniform(0.7, 1.5)
                        
                        hotel = HotelInfo(
                            hotel_id=place['fsq_id'],
                            name=place['name'],
                            price_per_night=round(base_price * price_variation),
                            rating=round(random.uniform(3.5, 4.8), 1),
                            amenities=self._generate_hotel_amenities(),
                            location=place.get('location', {}).get('address', 'City Center'),
                            distance_to_center=f"{place.get('distance', random.randint(500, 3000))}m",
                            currency="USD"
                        )
                        hotels.append(hotel)
                    
                    return hotels[:4]
                else:
                    logger.warning(f"Foursquare API error: {response.status}")
                    return []
        except Exception as e:
            logger.warning(f"Error calling Foursquare API: {e}")
            return []
    
    def _get_hotel_base_price(self, destination: str) -> float:
        """Get base hotel pricing for destination."""
        pricing_tiers = {
            'London': 250,
            'Paris': 200,
            'New York': 300,
            'Tokyo': 180,
            'Rome': 150
        }
        return pricing_tiers.get(destination, 120)
    
    def _generate_hotel_amenities(self) -> List[str]:
        """Generate realistic hotel amenities."""
        all_amenities = [
            'Free WiFi', 'Breakfast', 'Gym', 'Pool', 'Spa', 'Restaurant', 
            'Bar', 'Concierge', 'Room Service', 'Parking', 'Pet Friendly',
            'Business Center', 'Laundry', 'Airport Shuttle', 'Air Conditioning'
        ]
        return random.sample(all_amenities, k=random.randint(4, 8))
    
    def search_hotels_fallback(self, destination: str, check_in: str, check_out: str, 
                             travelers: int = 1) -> List[HotelInfo]:
        """Fallback hotel search with realistic data."""
        hotel_chains = {
            'London': ['The Langham London', 'Claridge\'s', 'The Savoy', 'Mandarin Oriental'],
            'Paris': ['Le Meurice', 'Hotel Plaza Athénée', 'The Ritz Paris', 'Hotel George V'],
            'New York': ['The Plaza', 'The St. Regis', 'The Carlyle', 'The Pierre'],
            'Tokyo': ['The Peninsula Tokyo', 'Mandarin Oriental Tokyo', 'The Ritz-Carlton Tokyo'],
            'Rome': ['Hotel de Russie', 'The St. Regis Rome', 'Hotel Artemide']
        }
        
        default_hotels = ['Grand Plaza Hotel', 'Luxury Suites', 'Premium Resort', 'City Center Hotel']
        
        hotel_names = hotel_chains.get(destination, default_hotels)
        base_price = self._get_hotel_base_price(destination)
        
        hotels = []
        for i in range(4):
            price_variation = random.uniform(0.6, 1.4)
            hotel = HotelInfo(
                hotel_id=f"HTL{random.randint(1000, 9999)}",
                name=random.choice(hotel_names),
                price_per_night=round(base_price * price_variation),
                rating=round(random.uniform(4.0, 4.9), 1),
                amenities=self._generate_hotel_amenities(),
                location=random.choice(['City Center', 'Downtown', 'Near Airport', 'Business District']),
                distance_to_center=f"{random.uniform(0.5, 5.0):.1f} km",
                currency="USD"
            )
            hotels.append(hotel)
        
        return hotels
    
    async def search_hotels(self, destination: str, check_in: str, check_out: str, 
                          travelers: int = 1) -> List[HotelInfo]:
        """Search hotels with API first, fallback to generated data."""
        # Try API first
        hotels = await self.search_hotels_api(destination, check_in, check_out, travelers)
        
        # Use fallback if API failed
        if not hotels:
            logger.info("Using fallback hotel data")
            hotels = self.search_hotels_fallback(destination, check_in, check_out, travelers)
        
        return hotels
    
    async def search_activities_api(self, destination: str, activity_type: Optional[str] = None) -> List[ActivityInfo]:
        """Search for activities using Foursquare and Eventbrite APIs."""
        activities = []
        
        # Get activities from Foursquare Places
        if config.api.foursquare_api_key:
            try:
                dest_info = config.travel.popular_destinations.get(destination)
                if dest_info:
                    lat, lon = dest_info["coordinates"]["lat"], dest_info["coordinates"]["lon"]
                    
                    # Search for tourist attractions
                    url = f"{config.api.foursquare_base_url}/search"
                    params = {
                        'll': f"{lat},{lon}",
                        'categories': '16000',  # Arts & Entertainment
                        'limit': 5
                    }
                    
                    headers = {
                        'Authorization': config.api.foursquare_api_key,
                        'Accept': 'application/json'
                    }
                    
                    async with self.session.get(url, params=params, headers=headers) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            for place in data.get('results', []):
                                activity = ActivityInfo(
                                    activity_id=place['fsq_id'],
                                    name=place['name'],
                                    category='attractions',
                                    price=random.randint(15, 80),
                                    duration=f"{random.randint(1, 4)} hours",
                                    rating=round(random.uniform(4.0, 4.8), 1),
                                    description=f"Popular attraction in {destination}",
                                    currency="USD"
                                )
                                activities.append(activity)
            except Exception as e:
                logger.warning(f"Error getting activities from Foursquare: {e}")
        
        # Get events from Eventbrite
        if config.api.eventbrite_token:
            try:
                url = f"{config.api.eventbrite_base_url}/events/search"
                params = {
                    'location.address': destination,
                    'expand': 'venue',
                    'sort_by': 'relevance'
                }
                
                headers = {
                    'Authorization': f'Bearer {config.api.eventbrite_token}'
                }
                
                async with self.session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        for event in data.get('events', [])[:3]:
                            if event.get('is_free'):
                                price = 0
                            else:
                                price = random.randint(25, 150)
                            
                            activity = ActivityInfo(
                                activity_id=event['id'],
                                name=event['name']['text'],
                                category='events',
                                price=price,
                                duration="3 hours",
                                rating=round(random.uniform(4.2, 4.9), 1),
                                description=event.get('description', {}).get('text', '')[:200] + "...",
                                currency="USD"
                            )
                            activities.append(activity)
            except Exception as e:
                logger.warning(f"Error getting events from Eventbrite: {e}")
        
        return activities
    
    def search_activities_fallback(self, destination: str, activity_type: Optional[str] = None) -> List[ActivityInfo]:
        """Fallback activities search with curated data."""
        activity_db = {
            'London': [
                {'name': 'Tower of London Tour', 'category': 'historical', 'price': 28, 'duration': '3 hours'},
                {'name': 'Westminster Abbey Visit', 'category': 'cultural', 'price': 24, 'duration': '2 hours'},
                {'name': 'London Eye Experience', 'category': 'sightseeing', 'price': 32, 'duration': '1 hour'},
                {'name': 'West End Theater Show', 'category': 'entertainment', 'price': 65, 'duration': '3 hours'},
                {'name': 'Thames River Cruise', 'category': 'sightseeing', 'price': 22, 'duration': '2 hours'},
                {'name': 'British Museum Tour', 'category': 'cultural', 'price': 20, 'duration': '3 hours'}
            ],
            'Paris': [
                {'name': 'Eiffel Tower Visit', 'category': 'sightseeing', 'price': 26, 'duration': '2 hours'},
                {'name': 'Louvre Museum Tour', 'category': 'cultural', 'price': 17, 'duration': '4 hours'},
                {'name': 'Seine River Cruise', 'category': 'sightseeing', 'price': 15, 'duration': '1 hour'},
                {'name': 'Versailles Palace Tour', 'category': 'historical', 'price': 20, 'duration': '6 hours'},
                {'name': 'Montmartre Walking Tour', 'category': 'cultural', 'price': 25, 'duration': '3 hours'},
                {'name': 'Cooking Class Experience', 'category': 'culinary', 'price': 120, 'duration': '4 hours'}
            ],
            'New York': [
                {'name': 'Statue of Liberty Tour', 'category': 'sightseeing', 'price': 23, 'duration': '4 hours'},
                {'name': 'Empire State Building', 'category': 'sightseeing', 'price': 37, 'duration': '2 hours'},
                {'name': 'Broadway Show', 'category': 'entertainment', 'price': 125, 'duration': '3 hours'},
                {'name': 'Central Park Tour', 'category': 'outdoor', 'price': 15, 'duration': '2 hours'},
                {'name': '9/11 Memorial Visit', 'category': 'historical', 'price': 26, 'duration': '2 hours'},
                {'name': 'Food Tour', 'category': 'culinary', 'price': 75, 'duration': '3 hours'}
            ]
        }
        
        default_activities = [
            {'name': 'City Walking Tour', 'category': 'sightseeing', 'price': 20, 'duration': '3 hours'},
            {'name': 'Local Museum Visit', 'category': 'cultural', 'price': 15, 'duration': '2 hours'},
            {'name': 'Food Tasting Tour', 'category': 'culinary', 'price': 60, 'duration': '3 hours'},
            {'name': 'Local Market Tour', 'category': 'cultural', 'price': 25, 'duration': '2 hours'}
        ]
        
        dest_activities = activity_db.get(destination, default_activities)
        
        activities = []
        selected_activities = random.sample(dest_activities, k=min(4, len(dest_activities)))
        
        for act_data in selected_activities:
            activity = ActivityInfo(
                activity_id=f"ACT{random.randint(1000, 9999)}",
                name=act_data['name'],
                category=act_data['category'],
                price=act_data['price'],
                duration=act_data['duration'],
                rating=round(random.uniform(4.0, 4.8), 1),
                description=f"Experience {act_data['name']} in {destination}",
                currency="USD"
            )
            activities.append(activity)
        
        return activities
    
    async def search_activities(self, destination: str, activity_type: Optional[str] = None) -> List[ActivityInfo]:
        """Search activities with API first, fallback to curated data."""
        # Try APIs first
        activities = await self.search_activities_api(destination, activity_type)
        
        # Use fallback if API failed or returned insufficient results
        if len(activities) < 3:
            logger.info("Using fallback activity data")
            fallback_activities = self.search_activities_fallback(destination, activity_type)
            activities.extend(fallback_activities)
        
        return activities[:6]  # Return max 6 activities
    
    async def get_exchange_rate(self, from_currency: str, to_currency: str) -> float:
        """Get exchange rate between currencies."""
        if not config.api.exchangerate_api_key or from_currency == to_currency:
            return 1.0
        
        try:
            url = f"{config.api.exchangerate_base_url}/{config.api.exchangerate_api_key}/pair/{from_currency}/{to_currency}"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('conversion_rate', 1.0)
                else:
                    logger.warning(f"Exchange rate API error: {response.status}")
                    return 1.0
        except Exception as e:
            logger.warning(f"Error getting exchange rate: {e}")
            return 1.0

# Synchronous wrapper functions for easier integration
def search_flights_sync(origin: str, destination: str, departure_date: str, travelers: int = 1) -> List[FlightInfo]:
    """Synchronous wrapper for flight search."""
    async def _search():
        async with DataSourceManager() as dsm:
            return await dsm.search_flights(origin, destination, departure_date, travelers)
    
    try:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(_search())
    except RuntimeError:
        # Create new event loop if none exists
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(_search())
        finally:
            loop.close()

def search_hotels_sync(destination: str, check_in: str, check_out: str, travelers: int = 1) -> List[HotelInfo]:
    """Synchronous wrapper for hotel search."""
    async def _search():
        async with DataSourceManager() as dsm:
            return await dsm.search_hotels(destination, check_in, check_out, travelers)
    
    try:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(_search())
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(_search())
        finally:
            loop.close()

def search_activities_sync(destination: str, activity_type: Optional[str] = None) -> List[ActivityInfo]:
    """Synchronous wrapper for activity search."""
    async def _search():
        async with DataSourceManager() as dsm:
            return await dsm.search_activities(destination, activity_type)
    
    try:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(_search())
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(_search())
        finally:
            loop.close()

def get_weather_sync(location: str, date: str) -> WeatherInfo:
    """Synchronous wrapper for weather data."""
    async def _get_weather():
        async with DataSourceManager() as dsm:
            return await dsm.get_weather(location, date)
    
    try:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(_get_weather())
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(_get_weather())
        finally:
            loop.close()