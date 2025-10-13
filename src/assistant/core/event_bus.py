import asyncio
from typing import Dict, List, Callable, Any
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class Event:
    """Standard event structure"""
    event_type: str
    timestamp: float
    payload: Any
    source: str
    correlation_id: str = None

class EventBus:
    """Central event bus for pub/sub communication"""

    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._running = False

    async def start(self):
        """Start event processing loop"""
        self._running = True
        asyncio.create_task(self._process_events())

    async def stop(self):
        """Stop event processing"""
        self._running = False

    def subscribe(self, event_type: str, handler: Callable):
        """Subscribe to event type"""
        self._subscribers[event_type].append(handler)

    def unsubscribe(self, event_type: str, handler: Callable):
        """Unsubscribe from event type"""
        if handler in self._subscribers[event_type]:
            self._subscribers[event_type].remove(handler)

    async def publish(self, event: Event):
        """Publish event to queue"""
        await self._event_queue.put(event)

    async def _process_events(self):
        """Process events from queue"""
        while self._running:
            try:
                event = await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=0.1
                )

                # Call all subscribers for this event type
                for handler in self._subscribers[event.event_type]:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(event)
                        else:
                            handler(event)
                    except Exception as e:
                        print(f"Error in event handler: {e}")

            except asyncio.TimeoutError:
                continue
