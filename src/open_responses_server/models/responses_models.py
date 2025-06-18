from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional

# Pydantic models for the /responses API endpoint

class ToolFunction(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict] = None

class Tool(BaseModel):
    type: str = "function"
    function: ToolFunction

class OutputText(BaseModel):
    type: str = "output_text"
    text: str

class Message(BaseModel):
    id: Optional[str] = None
    type: str = "message"
    role: str
    content: List[Any]

class TextFormat(BaseModel):
    type: str = "text"

class ResponseItem(BaseModel):
    id: str
    type: str
    role: str
    content: List[Any]

class ResponseModel(BaseModel):
    id: str
    object: str = "response"
    created_at: int
    status: str = "in_progress"
    error: Optional[Any] = None
    incomplete_details: Optional[Any] = None
    instructions: Optional[str] = None
    max_output_tokens: Optional[int] = None
    model: str
    output: List[Any] = []
    parallel_tool_calls: bool = True
    previous_response_id: Optional[str] = None
    reasoning: Dict = Field(default_factory=lambda: {"effort": None, "summary": None})
    store: bool = False
    temperature: float = 1.0
    text: Dict = Field(default_factory=lambda: {"format": {"type": "text"}})
    tool_choice: str = "auto"
    tools: List[Tool] = []
    top_p: float = 1.0
    truncation: str = "disabled"
    usage: Optional[Dict] = None
    user: Optional[str] = None
    metadata: Dict = Field(default_factory=dict)

class ResponseCreateRequest(BaseModel):
    model: str
    input: Optional[List[Any]] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[str] = "auto"
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    max_output_tokens: Optional[int] = None
    user: Optional[str] = None
    metadata: Optional[Dict] = None

# Models for streaming events
class ToolCallArgumentsDelta(BaseModel):
    type: str = "response.function_call_arguments.delta"
    item_id: str
    output_index: int
    delta: str

class ToolCallArgumentsDone(BaseModel):
    type: str = "response.function_call_arguments.done"
    id: str
    output_index: int
    arguments: str

class ToolCallsCreated(BaseModel):
    type: str = "response.tool_calls.created"
    item_id: str
    output_index: int
    tool_call: Dict

class OutputTextDelta(BaseModel):
    type: str = "response.output_text.delta"
    item_id: str
    output_index: int
    delta: str

class ResponseCreated(BaseModel):
    type: str = "response.created"
    response: ResponseModel

class ResponseInProgress(BaseModel):
    type: str = "response.in_progress"
    response: ResponseModel

class ResponseCompleted(BaseModel):
    type: str = "response.completed"
    response: ResponseModel 