import uuid
from datetime import datetime
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


# ── Enums ──────────────────────────────────────────────────────────────────────

class MessageRole(str, Enum):
    user      = "user"
    assistant = "assistant"
    system    = "system"
    tool      = "tool"


# ── Sub-models ─────────────────────────────────────────────────────────────────

class ScreenElement(BaseModel):
    """A single UI element detected by OmniParser (icon or text)."""
    id:           int
    type:         str            # "icon" | "text"
    content:      str = ""       # OCR text (empty for icons)
    bbox_pixel:   list[int]      # [x1, y1, x2, y2] in pixels
    center_pixel: list[int]      # [cx, cy] in pixels
    interactable: bool = True


class ScreenAnnotation(BaseModel):
    """OmniParser detection results attached to a screenshot message."""
    elements:     list[ScreenElement] = Field(default_factory=list)
    image_width:  int = 0
    image_height: int = 0
    latency_ms:   int = 0


class PreviousMessage(BaseModel):
    """A single turn in the conversation history."""
    role:        MessageRole
    content:     str
    annotations: Optional[ScreenAnnotation] = Field(
        default=None,
        description="OmniParser UI element detections (only on screenshot messages)"
    )
    # Tool calling support
    tool_calls:   Optional[list[dict]] = Field(
        default=None,
        description="OpenAI-format tool_calls (only on assistant messages with tool use)"
    )
    tool_call_id: Optional[str] = Field(
        default=None,
        description="ID of the tool call this message is a result for (only on tool role messages)"
    )
    tool_name: Optional[str] = Field(
        default=None,
        description="Name of the tool that was called (only on tool role messages)"
    )
    timestamp:   datetime = Field(default_factory=datetime.utcnow)


# ── Agent request ──────────────────────────────────────────────────────────────

class AgentRequest(BaseModel):
    """
    Payload sent to the VLM for each agent step.

    Required
    --------
    session_id       : Unique ID for the current user session. Allows the
                       backend to correlate steps and maintain state.
    user_message     : The natural-language instruction from the user.
    base64_screenshot: Current screen state encoded as a base64 PNG string.
    timestamp        : When this request was created (UTC).

    Conversation context
    --------------------
    previous_messages: Ordered list of prior turns (system prompt + history).
                       Empty list on the first turn of a session.

    Task / step metadata
    --------------------
    task_id          : Optional identifier for the high-level task this step
                       belongs to. Useful when one user message spawns multiple
                       agent steps.
    step_index       : Zero-based index of this step within the current task.
                       Helps the LLM understand how far along execution is.

    Screen metadata
    ---------------
    display_scale    : DPI scale factor (e.g. 1.5 for 150 % scaling). Lets
                       the backend normalise coordinates before acting.
    """

    # Core fields
    session_id:        Optional[str] = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique session identifier"
    )
    user_message:      str      = Field(..., description="Natural-language instruction from the user")
    base64_screenshot: str      = Field(default="", description="Current screen state as a base64-encoded PNG")
    timestamp:         datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC time this request was created"
    )

    # Conversation context
    previous_messages: list[PreviousMessage] = Field(
        default_factory=list,
        description="Ordered conversation history (system prompt + prior turns)"
    )

    # Task / step metadata
    task_id:    Optional[str] = Field(default=None, description="ID of the parent task")
    step_index: int           = Field(default=0, ge=0, description="Zero-based step index within the task")
    agent_mode: Literal["coworker", "remote"] = Field(
        default="coworker",
        description="Selected frontend collaboration mode"
    )

    # Origin of the request. "local" = in-app chat; "whatsapp"/"imessage" = inbound
    # via the messaging/ bridge runner. Used for logging and for the model prompt
    # so the agent can adapt its tone for off-host channels.
    source: Literal["local", "whatsapp", "imessage", "discord"] = Field(
        default="local",
        description="Transport that originated this request"
    )

    # Screen metadata
    display_scale: float = Field(
        default=1.0, gt=0,
        description="DPI scale factor (e.g. 1.5 = 150 % scaling)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                "user_message": "Open TextEdit and type hello world",
                "base64_screenshot": "<base64-encoded-png>",
                "timestamp": "2026-03-03T12:00:00Z",
                "previous_messages": [
                    {"role": "system",    "content": "You are a desktop automation agent."},
                    {"role": "assistant", "content": "Ready. What would you like me to do?"}
                ],
                "task_id": "task-001",
                "step_index": 0,
                "agent_mode": "coworker",
                "display_scale": 1.0
            }
        }


# ── Inbound HTTP request models ────────────────────────────────────────────────

class ScreenshotRequest(BaseModel):
    """Screenshot payload sent from Electron desktopCapturer."""
    image:  str       # raw base64-encoded PNG
    width:  int = 0
    height: int = 0


class ActionCompleteRequest(BaseModel):
    """Electron notifies the backend that a dispatched action has finished."""
    session_id:  str
    ipc_channel: str
    success:     bool
    error:       Optional[str] = None
    output:      Optional[str] = None


class StopRequest(BaseModel):
    """User requests the agent to stop the current trajectory."""
    session_id: str


class CompactRequest(BaseModel):
    """User requests context compaction for a bloated session."""
    session_id: str


class ContinueSessionRequest(BaseModel):
    """User wants to continue a previous session."""
    previous_session_id: str
    agent_mode: Literal["coworker", "remote"] = "coworker"


class ProviderSettingsRequest(BaseModel):
    """Update active provider, model, and API key."""
    provider: str
    model: str = ""
    api_key: str = ""
