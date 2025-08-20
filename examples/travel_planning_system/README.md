# Enhanced Multi-Agent Travel Planning System

A sophisticated travel planning system built with LangGraph that uses multiple specialized AI agents to create comprehensive travel plans. The system integrates with real APIs for authentic data while providing intelligent fallbacks.

## 🌟 Features

### Multi-Agent Architecture
- **Coordinator Agent**: Manages overall planning process and extracts requirements
- **Flight Specialist**: Searches real flight data using Amadeus API
- **Hotel Specialist**: Finds accommodations using Foursquare Places API  
- **Activity Specialist**: Discovers attractions and events using multiple APIs
- **Budget Analyst**: Provides detailed cost analysis and optimization
- **Plan Synthesizer**: Creates comprehensive final travel plans

### Real Data Integration
- **Weather**: OpenWeatherMap API for accurate forecasts
- **Flights**: Amadeus API for real flight search and pricing
- **Hotels**: Foursquare Places API for accommodation discovery
- **Activities**: Foursquare + Eventbrite APIs for attractions and events
- **Currency**: ExchangeRate API for international trip budgeting
- **Fallback Data**: Realistic generated data when APIs unavailable

### Advanced Orchestration
- **LangGraph Workflow**: Sophisticated state management and agent coordination
- **Smart Handoffs**: Agents pass context and collaborate seamlessly  
- **Error Handling**: Graceful degradation with fallback data sources
- **Tracing Support**: Full observability with OpenTelemetry and Azure insights

## 🚀 Quick Start

### 1. Setup
```bash
# Clone and navigate to the travel planning system
cd examples/travel_planning_system

# Run the setup script
python setup.py
```

### 2. Configuration
Edit the `.env` file with your API keys:

```bash
# Required: LLM Configuration
AZURE_OPENAI_API_KEY=your_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/

# Optional: Real Data APIs (free tiers available)
OPENWEATHER_API_KEY=your_key_here
AMADEUS_API_KEY=your_key_here
AMADEUS_API_SECRET=your_secret_here
EXCHANGERATE_API_KEY=your_key_here
FOURSQUARE_API_KEY=your_key_here
EVENTBRITE_TOKEN=your_token_here
```

### 3. Run
```bash
python enhanced_multi_agent_system.py
```

## 📝 Sample Inputs

The system handles various complex travel scenarios with detailed requirements. Each sample demonstrates how comprehensive inputs trigger all six specialized agents and their tools:

### Business Trip
```
"I need to plan a comprehensive business trip to London, UK for 3 days from March 15-17, 2024. I'll be traveling from Seattle, Washington (SEA airport) and need to arrive by 9 AM London time on March 15th for an important board meeting at 11 AM. I prefer business class or premium economy flights with major airlines like British Airways, Virgin Atlantic, or Delta, and I need to return by 6 PM on March 17th to catch a connecting domestic flight.

For accommodations, I require a 4-star or 5-star hotel in or very close to the City of London financial district, within walking distance of Bank station or Monument station. The hotel must have business center facilities, high-speed WiFi, room service available until midnight, and preferably a concierge service. I'll need a quiet room (away from street noise) with a workspace, and the hotel should offer express check-in/check-out services. Important amenities include 24-hour fitness center, laundry services, and ideally a business lounge for informal meetings.

During my stay, I have specific business requirements: I need to visit the London Stock Exchange area on March 15th afternoon, attend client meetings in Canary Wharf on March 16th morning (so transportation connections are important), and have a business dinner planned near Covent Garden on March 16th evening. On March 17th, I have a morning meeting in Mayfair before my departure. I'll need reliable transportation between these locations and would prefer a mix of Underground and taxi services depending on timing and weather.

I also want to maximize my limited free time by visiting 1-2 quintessentially London cultural sites - perhaps Westminster Abbey, Tower Bridge, or a quick visit to the British Museum if time permits. I prefer efficiency over sightseeing, but would appreciate recommendations for excellent traditional British cuisine for my free meals. I'm particularly interested in historic pubs with character for casual meals and upscale restaurants for any business entertainment needs.

Specific travel requirements include: ensuring my mobile phone works seamlessly in the UK (need data for maps and communication), understanding currency exchange rates and preferred payment methods for business expenses, and knowing about tipping customs in restaurants and for services. I'll need to pack appropriate business attire for both indoor meetings and potentially walking outside between locations, so weather forecasts are crucial.

My total budget is approximately $3,000 USD, which should cover flights, accommodation, local transportation, meals, and any cultural activities. I need current weather forecasts for appropriate business attire packing, and I'd appreciate guidance on transportation options between the financial district and Canary Wharf. Please also include information about time zone considerations for my meetings, any important business etiquette or cultural notes for conducting business in London, emergency contact information for medical or business issues, and recommendations for extending the trip by one day if meetings run late. I also need advice on the best way to expense and track costs for corporate reimbursement, preferred business districts for evening networking if opportunities arise, and any seasonal considerations for March travel in London including potential weather delays that could affect my schedule."
```

### Family Vacation
```
"Please plan a comprehensive family vacation to Paris, France for our family of four - 2 adults (ages 38 and 42) and 2 children (ages 8 and 12) - for 10 days from July 20-29, 2024. We're flying from Los Angeles (LAX) and would prefer direct flights or flights with minimal layovers, departing in the morning if possible to maximize our first day. We're flexible on airline but prefer major carriers with good family services, entertainment systems for kids, and the option to select seats together.

For accommodations, we need a family-friendly hotel or apartment rental that can comfortably sleep four people - either connecting rooms, a suite, or a vacation rental with at least 2 bedrooms. Location is crucial: we want to be within easy metro access to major attractions but in a safe, family-friendly neighborhood. The 6th or 7th arrondissement would be ideal, or near the Latin Quarter. We need reliable WiFi, air conditioning (important for July weather), and ideally access to laundry facilities for a 10-day stay with children.

Our family loves museums and cultural experiences, but we need age-appropriate activities that will engage both an 8-year-old and a 12-year-old. Priority attractions include the Eiffel Tower (we want to go up it, preferably at sunset), the Louvre (but with a family-friendly tour), Notre-Dame area, and Versailles (we're willing to take a day trip). We're also very interested in Paris parks - Luxembourg Gardens, Tuileries, and especially areas where kids can play and run around.

Food is important to us - we want to experience authentic French cuisine but also need kid-friendly options. Our 8-year-old is somewhat picky, so we need recommendations for restaurants that cater to families, plus information about French foods kids typically enjoy. We'd love to try a Seine river cruise with dinner if it's family-appropriate, and we're interested in visiting local bakeries and markets where kids can sample treats.

We're interested in cultural activities like a family-friendly cooking class, visiting local markets (especially ones with samples!), and perhaps a short day trip outside Paris - maybe to Giverny to see Monet's gardens or to a nearby château that's good for kids. We want to balance educational experiences with fun activities that won't overwhelm young attention spans.

Transportation is a concern with children and luggage - we need advice on navigating the Paris metro system with kids, whether day passes or individual tickets work better for families, and alternative transportation options when metro stations don't have elevators. We'll also need recommendations for child-safe walking routes between major attractions.

Practical family considerations include: finding pharmacies that carry children's medications, understanding French emergency services and pediatric care if needed, locating playgrounds and family restrooms throughout the city, and knowing about child discounts for attractions and transportation. We'd also appreciate advice on managing jet lag with children and maintaining routines while traveling.

Our budget is flexible but approximately $8,000 total for the family, covering flights, accommodation, local transportation, meals, attraction tickets, and activities. We'd like current weather forecasts for packing (especially for walking with kids), recommendations for navigating Paris metro with children and luggage, and tips for keeping kids engaged during cultural site visits. Please also include information about French customs and etiquette when traveling with children, emergency contact information, and any special considerations for traveling to France with minors, including required documentation. We're also interested in age-appropriate souvenirs and educational materials we could purchase to help the kids remember and learn from their French cultural experience."
```

### Romantic Getaway
```
"I'm planning a special romantic anniversary trip to Rome, Italy for my partner and me (both in our early 30s) to celebrate our 5th wedding anniversary from September 10-16, 2024. This is a once-in-a-lifetime trip for us, so we want it to be absolutely perfect and memorable. We're departing from Chicago (ORD) and are willing to pay extra for premium accommodations and experiences.

For flights, we'd prefer business class or premium economy if budget allows, with a preference for evening departures so we can work a half day before leaving. We'd like to arrive in Rome refreshed and ready to start our romantic getaway, so flight comfort is important to us. We're flexible on airlines but prefer those known for excellent service and comfortable seating for long international flights.

We want luxury accommodations - a 5-star boutique hotel or a romantic bed & breakfast in the heart of Rome, preferably with historic character and charm. Location preferences include near the Spanish Steps, Trastevere for its romantic atmosphere, or close to the Pantheon area. We'd love a room with a view (either city views or a charming courtyard), luxury amenities like a marble bathroom, high-quality linens, and ideally room service for romantic in-room breakfasts. A rooftop terrace or balcony would be amazing for sunset aperitivos.

Our romantic wishlist includes classic Rome experiences but with elevated, intimate touches: a private or small-group tour of the Vatican Museums and Sistine Chapel early in the morning before crowds arrive, a sunset visit to the Trevi Fountain, and a romantic evening stroll through the cobblestone streets of Trastevere. We want to visit the Colosseum but prefer a special access tour, perhaps at sunset or with underground access.

For dining, we're serious food lovers who want authentic, high-end Roman cuisine. We'd love recommendations for romantic restaurants with exceptional cacio e pepe, carbonara, and other Roman specialties - places with intimate lighting, excellent wine lists, and preferably outdoor seating for warm September evenings. We're interested in a wine tasting experience, perhaps in Frascati or another nearby wine region, and would love to take a cooking class together focusing on traditional Roman dishes.

We're also interested in romantic day trips - perhaps to the charming hill towns of Orvieto or Civita di Bagnoregio, or to Villa d'Este in Tivoli for its romantic gardens. We love art and history but want experiences that feel special and intimate rather than crowded tourist activities. We'd be interested in a private tour of lesser-known churches with beautiful art, or a sunset photography session at romantic locations around the city.

Transportation preferences include comfortable private transfers from the airport, and for day trips, we'd prefer private drivers or small group tours rather than large bus tours. Within Rome, we're happy to walk for the romantic atmosphere but would like recommendations for the most scenic routes between attractions.

Special romantic touches we're considering: a professional photography session to capture our anniversary celebration, perhaps arranged flowers or champagne in our room, couples spa treatments, and reservations at restaurants known for marriage proposals (we're not proposing, but love the romantic atmosphere). We'd also appreciate recommendations for romantic shopping - perhaps for matching jewelry or art pieces to commemorate our trip.

Our budget is up to $6,000 for this special celebration, and we're willing to splurge on unique experiences, exceptional dining, and luxury accommodations. We'd appreciate current weather forecasts for September in Rome, recommendations for romantic photography spots, tips for avoiding crowds at major attractions, and suggestions for surprise romantic touches we could arrange for each other during the trip. Please also include information about Italian dining customs, tipping etiquette, and any special romantic traditions or experiences unique to Rome. We're also interested in learning a few romantic Italian phrases and understanding cultural customs around romance and relationships in Italy."
```

### Group Adventure
```
"I'm organizing an epic group adventure trip to Tokyo, Japan for 6 close friends (ages 25-32, mix of 4 males and 2 females) for 2 weeks from October 5-18, 2024. We're all adventurous travelers who love immersive cultural experiences, amazing food, and want to see both traditional Japan and ultra-modern Tokyo. We're departing from New York City (JFK) and are looking for the best value flights while still being comfortable for the long journey.

For accommodations, we need housing for 6 people and are open to creative solutions - either multiple hotel rooms (preferably 3 double rooms close to each other), a large vacation rental, or even a traditional ryokan experience for part of our stay. We want to stay in different neighborhoods to experience various aspects of Tokyo - perhaps Shibuya or Harajuku for the modern, energetic vibe, Asakusa for traditional culture, and maybe one night in a ryokan in nearby Hakone or Nikko for the authentic Japanese experience.

Our group is fascinated by Japanese culture and wants deep, authentic experiences. We're interested in traditional activities like a tea ceremony, visiting ancient temples and shrines (Senso-ji, Meiji Shrine), experiencing a traditional onsen, and learning about samurai culture. We'd love to visit a sumo training stable if possible, try our hand at traditional crafts like pottery or calligraphy, and explore traditional gardens like the East Gardens of the Imperial Palace.

At the same time, we're excited about modern Japan - we want to experience the crazy energy of Shibuya crossing, explore the gaming and anime culture in Akihabara, visit teamLab Borderless or a similar digital art experience, check out the Robot Restaurant or similar unique Tokyo entertainment, and explore the fashion and youth culture in Harajuku.

Food is a HUGE priority for our group - we're all serious foodies who want to experience the full spectrum of Japanese cuisine. We want to try authentic sushi at Tsukiji Outer Market, experience a traditional kaiseki dinner, find the best ramen shops (different styles in different neighborhoods), try authentic yakitori, and if budget allows, maybe even splurge on a meal at a Michelin-starred restaurant. We'd love to take a cooking class together focusing on Japanese techniques, visit a sake brewery or at least do a comprehensive sake tasting, and explore food markets and department store food courts.

We're also interested in adventure activities - hiking Mount Fuji if weather permits (we're visiting in October), visiting the snow monkeys in Jigokudani if we can fit in a day trip, exploring Nikko for its natural beauty and temples, or taking a day trip to Kamakura for its giant Buddha and hiking trails.

Our nightlife interests include experiencing Tokyo's unique bar culture - tiny bars in Golden Gai, karaoke (definitely want multiple karaoke sessions), and experiencing Tokyo's nightlife districts like Roppongi or Shinjuku. We'd love recommendations for bars where foreigners are welcome and can interact with locals.

Each person has a budget of approximately $4,000, so our total group budget is around $24,000, covering flights, accommodations, local transportation (including any day trips), meals, activities, and shopping. We want to experience efficient Japanese transportation - including rides on different types of trains, maybe even a short ride on the shinkansen bullet train to a nearby city.

Please provide current weather forecasts for October in Tokyo, comprehensive information about navigation and transportation for a group of 6 (including JR Pass recommendations), cultural etiquette and customs we should know, language basics for essential interactions, and any group activity recommendations that would be perfect for 6 friends traveling together. We'd also appreciate information about shopping districts, Japanese gift ideas to bring home, and any seasonal activities or festivals happening during our October visit."
```

### Cultural Exploration
```
"My partner and I (both culture enthusiasts in our late 40s with graduate degrees in art history and literature) want to plan an intensive cultural exploration of New York City for 5 days from November 8-12, 2024. We're flying from Denver and consider ourselves serious cultural travelers who prefer depth over breadth - we'd rather spend substantial time at fewer institutions than rush through many attractions.

We need accommodations in Manhattan that put us within easy reach of the major cultural district - ideally Midtown near the Museum Mile, or the Upper West Side near Lincoln Center. We prefer boutique hotels with character and history, sophisticated decor, and a location that feels quintessentially New York. A hotel with a good library or reading area would be perfect for our evening unwinding.

Our primary focus is New York's world-class museums, but we want to approach them strategically with expert guidance. At the Metropolitan Museum of Art, we want to spend an entire day with a private or semi-private expert guide focusing on specific periods - we're particularly interested in the Egyptian collection, European paintings, and the new modern wing. We plan to visit the Museum of Modern Art (MoMA) with emphasis on understanding the evolution of modern art movements, and we'd love insider access or a curator-led experience if possible.

For the Guggenheim, we're interested in both the current exhibitions and understanding Frank Lloyd Wright's architectural vision for the building itself. We want to visit the Frick Collection for its intimate setting and exceptional Old Masters, and if time permits, we're curious about some of the smaller, more specialized museums like the Morgan Library & Museum for its literary manuscripts and medieval artifacts.

Theater is essential to our New York experience - we want to see both a Broadway production (preferably something critically acclaimed rather than just popular) and an off-Broadway show that represents innovative contemporary theater. We'd appreciate recommendations based on current productions during our November visit, and we're willing to pay premium prices for excellent seats.

We're serious food enthusiasts who want to experience New York's culinary culture at the highest level. We're interested in restaurants that represent the pinnacle of various cuisines - perhaps a tasting menu at a Michelin-starred restaurant, authentic ethnic cuisine in neighborhoods like Chinatown or Little Italy, and iconic New York food experiences like a proper New York deli, a classic steakhouse, or an authentic New York pizza experience that locals would recommend.

For art galleries, we want to explore Chelsea's gallery district and SoHo, ideally with some guidance about current exhibitions and emerging artists. We're interested in understanding New York's contemporary art scene and would love recommendations for galleries showing cutting-edge work.

Architecture is another passion - we'd love a guided architectural walking tour that covers different periods of New York's development, from historic landmarks to modern skyscrapers. We're particularly interested in Art Deco buildings, the High Line's integration of urban planning and art, and understanding how New York's architecture reflects its cultural evolution.

We're also interested in literary New York - places associated with famous writers, historic literary venues, independent bookstores with character and knowledgeable staff, and perhaps a literary walking tour if one exists that goes beyond surface-level tourist information.

Our budget is approximately $3,500 total for both of us, covering flights, accommodation, meals, theater tickets, museum admissions, tours, and cultural activities. We prefer spending more on fewer, higher-quality experiences rather than trying to see everything. We'd appreciate information about museum memberships or cultural passes that might provide value and special access, recommendations for cultural events or lectures happening during our visit, and guidance on timing our museum visits to avoid crowds while maximizing our cultural immersion. We'd also love suggestions for sophisticated evening activities beyond theater - perhaps classical music performances at Lincoln Center, jazz clubs with authentic atmosphere, or cultural salons or intellectual gatherings if any are accessible to visitors."
```

## 🔧 API Setup Instructions

### Free API Keys (No Credit Card Required)

#### 1. OpenWeatherMap (Weather Data)
- Visit: https://openweathermap.org/api
- Sign up for free account
- Get API key (1000 calls/day free)
- Add to `.env`: `OPENWEATHER_API_KEY=your_key`

#### 2. Amadeus (Flight Data)
- Visit: https://developers.amadeus.com
- Create free developer account
- Get API key and secret (2000 calls/month free)
- Add to `.env`: 
  ```
  AMADEUS_API_KEY=your_key
  AMADEUS_API_SECRET=your_secret
  ```

#### 3. ExchangeRate API (Currency Data)
- Visit: https://exchangerate-api.com
- Sign up for free account
- Get API key (1500 calls/month free)
- Add to `.env`: `EXCHANGERATE_API_KEY=your_key`

#### 4. Foursquare Places (Hotels & Activities)
- Visit: https://developer.foursquare.com
- Create developer account
- Get API key (100 calls/day free)
- Add to `.env`: `FOURSQUARE_API_KEY=your_key`

#### 5. Eventbrite (Events Data)
- Visit: https://www.eventbrite.com/platform/api
- Create account and app
- Get private token (free)
- Add to `.env`: `EVENTBRITE_TOKEN=your_token`

### LLM Configuration

#### Azure OpenAI (Recommended)
```bash
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_DEPLOYMENT_NAME=gpt-4
```

#### OpenAI (Alternative)
```bash
OPENAI_API_KEY=your_key
```

## 🏗️ Architecture

### LangGraph Workflow
```
Start → Coordinator → Flight Specialist → Hotel Specialist → Activity Specialist → Budget Analyst → Plan Synthesizer → End
```

### State Management
The system maintains a comprehensive state object that tracks:
- User requirements and preferences
- Search results from each specialist
- Agent handoffs and collaboration history
- Final plan synthesis

### Agent Collaboration
Each agent:
1. Receives context from previous agents
2. Performs specialized tasks using real APIs
3. Updates shared state with findings
4. Hands off to next agent with recommendations

## 🔍 Tracing and Observability

### OpenTelemetry Support
- Comprehensive span tracking for all agent activities
- Tool usage and API call monitoring
- Performance metrics and error tracking

### Azure Application Insights Integration
- Production-ready observability
- GenAI semantic conventions compliance
- Content recording with PII protection

### Configuration
```bash
# Azure Application Insights
APPLICATION_INSIGHTS_CONNECTION_STRING=your_connection_string

# OpenTelemetry (local development)
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318/v1/traces
```

## 🛠️ Development

### Project Structure
```
travel_planning_system/
├── enhanced_multi_agent_system.py  # Main system
├── config.py                       # Configuration management
├── data_sources.py                 # API integrations
├── otel_tracer.py                  # OpenTelemetry tracer
├── setup.py                        # Setup and validation
├── requirements.txt                # Dependencies
├── .env.template                   # Configuration template
└── README.md                       # This file
```

### Key Components

#### Configuration (`config.py`)
- Centralized configuration management
- API key validation
- Environment-based settings
- Missing configuration detection

#### Data Sources (`data_sources.py`)
- Async API clients for all external services
- Intelligent fallback mechanisms
- Rate limiting and error handling
- Realistic data generation

#### Enhanced System (`enhanced_multi_agent_system.py`)
- LangGraph workflow implementation
- Specialized agent nodes
- State management and routing
- Comprehensive error handling

### Adding New APIs

1. Add configuration to `config.py`
2. Implement client in `data_sources.py`
3. Create fallback data generation
4. Update agent tools as needed
5. Test with and without API keys

## 🧪 Testing

### Manual Testing
```bash
# Test with sample inputs
python enhanced_multi_agent_system.py

# Test configuration
python setup.py
```

### System Validation
The setup script validates:
- Python version compatibility
- Required dependencies
- Configuration completeness
- Basic system functionality

## 📊 Performance

### API Usage Optimization
- Intelligent caching for repeated requests
- Batch API calls where possible
- Graceful degradation to fallbacks
- Rate limiting respect

### Resource Management
- Async HTTP client pooling
- Memory-efficient state management
- Configurable timeout handling
- Clean error recovery

## 🔒 Security and Privacy

### API Key Management
- Environment variable storage
- No hardcoded credentials
- Template-based configuration
- Validation warnings

### Content Recording
- Configurable PII protection
- Tracing content redaction
- Development vs production modes
- GDPR compliance considerations

## 🤝 Contributing

### Enhancement Ideas
- Additional destination coverage
- More specialized agents (visa, insurance)
- Integration with booking platforms
- Mobile app interface
- Voice interaction support

### Code Contributions
1. Follow existing patterns
2. Add comprehensive error handling
3. Include fallback mechanisms
4. Update documentation
5. Test with and without API keys

## 📄 License

This project is part of the LangGraph repository and follows the same licensing terms.

## 🆘 Support

### Common Issues

#### "No LLM configuration found"
- Add Azure OpenAI or OpenAI credentials to `.env`
- Run `python setup.py` to validate configuration

#### "API key missing" warnings
- Optional - system works with fallback data
- Add desired API keys for real data
- See API setup instructions above

#### Import errors
- Run `python setup.py` to install dependencies
- Ensure Python 3.8+ is installed

### Getting Help
1. Check the setup script output
2. Validate `.env` configuration
3. Review error messages for missing dependencies
4. Test with sample inputs first

---

**🎯 Goal**: Demonstrate sophisticated multi-agent orchestration with real data integration while maintaining usability and reliability.