from flask import Flask, request, jsonify
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# Define the tools for the agent
@tool
def recommend_destination(preferences: str) -> str:
    """Recommend travel destinations based on user preferences."""
    if "beach" in preferences.lower():
        return "Consider visiting Bali, Indonesia or the Maldives."
    elif "mountain" in preferences.lower():
        return "Consider visiting the Swiss Alps or the Rocky Mountains."
    elif "city" in preferences.lower():
        return "Consider visiting Paris, France or Tokyo, Japan."
    else:
        return "Consider visiting multiple destinations like Italy, Japan, or New Zealand."

@tool
def find_accommodation(location: str) -> str:
    """Find accommodations in a specific location."""
    if "bali" in location.lower():
        return "Top accommodations in Bali: Four Seasons Resort, Ayana Resort, and COMO Shambhala Estate."
    elif "paris" in location.lower():
        return "Top accommodations in Paris: Hotel Ritz Paris, Le Bristol, and Shangri-La Hotel."
    else:
        return "Check platforms like Booking.com or Airbnb for accommodations in your desired location."

@tool
def plan_activities(destination: str) -> str:
    """Recommend activities or attractions in a destination."""
    if "bali" in destination.lower():
        return "Top activities in Bali: Visit Uluwatu Temple, explore Ubud Monkey Forest, and relax at Nusa Dua Beach."
    elif "paris" in destination.lower():
        return "Top activities in Paris: Visit the Eiffel Tower, explore the Louvre Museum, and take a Seine River cruise."
    else:
        return "Check TripAdvisor or local guides for activities in your destination."

@tool
def book_flights(destination: str, travel_dates: str) -> str:
    """Search for flights to a destination."""
    return f"Flight options to {destination} on {travel_dates}: Economy from $500, Business from $1200."

@tool
def check_weather(destination: str, date: str) -> str:
    """Provide weather information for a destination."""
    if "bali" in destination.lower():
        return f"Weather in Bali on {date}: 30°C, sunny with occasional rain."
    elif "paris" in destination.lower():
        return f"Weather in Paris on {date}: 15°C, partly cloudy."
    else:
        return f"Check weather.com for the latest forecast in {destination}."

@tool
def plan_budget(destination: str, duration: int, preferences: str) -> str:
    """Estimate the cost of a trip based on user preferences."""
    if "luxury" in preferences.lower():
        return f"Estimated cost for a {duration}-day luxury trip to {destination}: $5000."
    elif "budget" in preferences.lower():
        return f"Estimated cost for a {duration}-day budget trip to {destination}: $1500."
    else:
        return f"Estimated cost for a {duration}-day trip to {destination}: $3000."

@tool
def recommend_cuisine(destination: str) -> str:
    """Suggest local dishes and restaurants in a destination."""
    if "bali" in destination.lower():
        return "Top dishes in Bali: Nasi Goreng, Babi Guling, and Bebek Betutu. Recommended restaurants: Locavore, Mozaic."
    elif "paris" in destination.lower():
        return "Top dishes in Paris: Croissant, Coq au Vin, and Escargot. Recommended restaurants: Le Jules Verne, L'Ambroisie."
    else:
        return "Check local guides for cuisine recommendations in your destination."

# Set up the models (simulated for DeepSeek-Reasoner and DeepSeek-Chat)
class DeepSeekReasoner:
    def invoke(self, input: str) -> str:
        return f"DeepSeek-Reasoner: {input}"

class DeepSeekChat:
    def invoke(self, input: str) -> str:
        return f"DeepSeek-Chat: {input}"

# Create the agent
tools = [
    recommend_destination,
    find_accommodation,
    plan_activities,
    book_flights,
    check_weather,
    plan_budget,
    recommend_cuisine
]
reasoner = DeepSeekReasoner()
checkpointer = MemorySaver()
app = create_react_agent(reasoner, tools, checkpointer=checkpointer)

# Deploy the agent using Flask
flask_app = Flask(__name__)

@flask_app.route('/travel', methods=['POST'])
def travel():
    user_input = request.json.get('message')
    final_state = app.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config={"configurable": {"thread_id": 42}}
    )
    return jsonify({"response": final_state["messages"][-1].content})

if __name__ == '__main__':
    flask_app.run(host='0.0.0.0', port=53186)