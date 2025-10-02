import pytest
from multimodal_assistant.core.event_bus import EventBus, Event
import asyncio

@pytest.mark.asyncio
async def test_event_bus():
    """Test event bus pub/sub"""
    bus = EventBus()
    await bus.start()

    received_events = []

    async def handler(event: Event):
        received_events.append(event)

    bus.subscribe("test_event", handler)

    await bus.publish(Event(
        event_type="test_event",
        timestamp=0.0,
        payload={"data": "test"},
        source="test"
    ))

    # Give time for event processing
    await asyncio.sleep(0.2)

    assert len(received_events) == 1
    assert received_events[0].event_type == "test_event"
    assert received_events[0].payload["data"] == "test"

    await bus.stop()

@pytest.mark.asyncio
async def test_event_bus_multiple_subscribers():
    """Test multiple subscribers to same event"""
    bus = EventBus()
    await bus.start()

    received_1 = []
    received_2 = []

    async def handler_1(event: Event):
        received_1.append(event)

    async def handler_2(event: Event):
        received_2.append(event)

    bus.subscribe("test_event", handler_1)
    bus.subscribe("test_event", handler_2)

    await bus.publish(Event(
        event_type="test_event",
        timestamp=0.0,
        payload={},
        source="test"
    ))

    await asyncio.sleep(0.2)

    assert len(received_1) == 1
    assert len(received_2) == 1

    await bus.stop()
