"""
Routing for the automation API endpoints
"""
from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import json
import os
from models import NaturalLanguageRequest, AutomationResponse

router = APIRouter(tags=["automation"])

# Load dummy data for testing


def load_dummy_data() -> Dict[str, Any]:
    """Load dummy actions from JSON file"""
    dummy_file = os.path.join(os.path.dirname(
        os.path.dirname(__file__)), "dummy_data.json")
    try:
        with open(dummy_file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        # If file not found, return a basic test sequence
        return {
            "actions": [
                {
                    "name": "Open Google",
                    "type": "newtab",
                    "url": "https://www.google.com"
                },
                {
                    "name": "Wait for page to load",
                    "type": "wait",
                    "value": "2000"
                }
            ]
        }


@router.post("/process", response_model=AutomationResponse)
async def process_request(request: NaturalLanguageRequest):
    try:
        # Import the agent function from services
        from services.agent import agent

        # Generate a thread ID based on timestamp
        import time
        thread_id = f"thread_{int(time.time())}"

        # Call the agent with the natural language request
        result = agent(request.query, thread_id=thread_id)

        # Check if we have a valid plan
        if result.get("plan") and isinstance(result["plan"], dict) and "plan" in result["plan"]:
            # Extract the actions from the plan
            actions = result["plan"]["plan"]
            return {"actions": actions}
        elif result.get("plan") and isinstance(result["plan"], dict):
            # If there's no nested "plan" key but there are actions in the root
            return {"actions": result["plan"].get("actions", [])}
        else:
            # Fallback to dummy data if no valid plan
            print(
                f"No valid plan found in agent response, using dummy data. Result: {result}")
            return load_dummy_data()

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing request: {str(e)}")
