from brokle import Brokle, observe, get_client
import os

# Set environment variables for get_client() used by decorator
os.environ["BROKLE_API_KEY"] = "bk_SZJvBQDr9brY80Ln1ceNtGZMoSNc175rs3gXbnLK"
os.environ["BROKLE_BASE_URL"] = "http://localhost:8080"
os.environ["BROKLE_ENVIRONMENT"] = "test"

# Initialize the singleton client that decorator will use
client = get_client()

@observe(capture_input=True, capture_output=True)
def get_weather(location: str, units: str = "celsius"):
    return {"temp": 25, "location": location, "units": units}

result = get_weather("Bangalore", units="fahrenheit")
client.flush()
print(f"✅ Result: {result}")
print(f"✅ Check traces table for input/output!")
