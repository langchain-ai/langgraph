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

The system handles various travel scenarios:

### Business Trip
```
"I need to plan a business trip to London for 3 days. I'll be traveling from Seattle, need a hotel near the financial district, and prefer efficient flights. Budget is around $3000."
```

### Family Vacation
```
"Plan a family vacation to Paris for 2 adults and 2 children for 10 days in summer. We love museums, parks, and family-friendly activities. Budget is flexible but around $8000 total."
```

### Romantic Getaway
```
"Plan a romantic anniversary trip to Rome for 2 people for a week. We want luxury accommodations, fine dining, and romantic experiences. Budget up to $6000."
```

### Group Adventure
```
"Organize a group trip to Tokyo for 6 friends for 2 weeks. We're interested in both traditional culture and modern experiences, food tours, and adventure activities. Budget $4000 per person."
```

### Cultural Exploration
```
"I want to explore New York for cultural experiences - museums, theaters, art galleries, and culinary tours. 5 days for 2 people, mid-range budget around $3500 total."
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