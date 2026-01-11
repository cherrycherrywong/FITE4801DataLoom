import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

# Use environment variables when available
API_KEY = os.getenv('BASE44_API_KEY', "8232d62b6cf24e7eb74c7f765abb6e10")
APP_ID = os.getenv('BASE44_APP_ID', "69160892745be891ce4c021a")
URL = os.getenv('BASE44_ENDPOINT_URL', f"https://app.base44.com/api/apps/{APP_ID}/entities/TravelDataProfile")

def download_data():
    headers = {'api_key': API_KEY, 'Content-Type': 'application/json'}
    response = requests.get(URL, headers=headers)
    
    if response.status_code == 200:
        with open('raw_travel_data.json', 'w') as f:
            json.dump(response.json(), f)
        print("Successfully extracted data to raw_travel_data.json")
    else:
        print(f"Error: {response.status_code}")

if __name__ == "__main__":
    download_data()