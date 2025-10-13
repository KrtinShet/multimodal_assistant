from enum import Enum
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import asyncio

class PipelineState(Enum):
    """Pipeline state machine"""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    RESPONDING = "responding"

@dataclass
class SessionState:
    """Session state container"""
    session_id: str
    conversation_history: list = field(default_factory=list)
    user_context: Dict[str, Any] = field(default_factory=dict)
    current_phase: PipelineState = PipelineState.IDLE

class StateManager:
    """Manages application state"""

    def __init__(self):
        self._current_session: Optional[SessionState] = None
        self._lock = asyncio.Lock()

    async def create_session(self, session_id: str) -> SessionState:
        """Create new session"""
        async with self._lock:
            self._current_session = SessionState(session_id=session_id)
            return self._current_session

    async def get_current_session(self) -> Optional[SessionState]:
        """Get current session"""
        return self._current_session

    async def update_phase(self, phase: PipelineState):
        """Update pipeline phase"""
        async with self._lock:
            if self._current_session:
                self._current_session.current_phase = phase

    async def add_message(self, role: str, content: str):
        """Add message to conversation history"""
        async with self._lock:
            if self._current_session:
                self._current_session.conversation_history.append({
                    "role": role,
                    "content": content,
                    "timestamp": asyncio.get_event_loop().time()
                })

    async def close_session(self):
        """Close current session"""
        async with self._lock:
            self._current_session = None
