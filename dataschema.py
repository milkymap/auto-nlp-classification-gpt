
from enum import Enum 
from pydantic import BaseModel

from typing import List, Dict, Any, Optional, Tuple, Awaitable

class Role(str, Enum):
    USER: str = "user"
    SYSTEM: str = "system"
    ASSISTANT: str = "assistant"

class Message(BaseModel):
    role: Role
    content:str 

class PredictionReqModel(BaseModel):
    text: str 
