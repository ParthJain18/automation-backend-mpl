from pydantic import BaseModel
from typing import List, Dict, Any, Optional


class NaturalLanguageRequest(BaseModel):
    """Model for natural language automation request"""
    query: str


class AutomationAction(BaseModel):
    """Model for a single automation action"""
    name: str
    type: str
    selector: Optional[str] = None
    value: Optional[str] = None
    url: Optional[str] = None


class AutomationResponse(BaseModel):
    """Model for the response containing automation actions"""
    actions: List[Dict[str, Any]]
