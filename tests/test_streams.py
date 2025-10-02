import pytest
from multimodal_assistant.core.streams import AsyncStream

@pytest.mark.asyncio
async def test_stream_basic():
    """Test basic stream operations"""
    stream = AsyncStream[int]()
    await stream.put(1)
    await stream.put(2)
    await stream.close()

    assert await stream.next() == 1
    assert await stream.next() == 2
    assert await stream.next() is None

@pytest.mark.asyncio
async def test_stream_iteration():
    """Test async iteration over stream"""
    stream = AsyncStream[str]()

    async def producer():
        await stream.put("hello")
        await stream.put("world")
        await stream.close()

    import asyncio
    asyncio.create_task(producer())

    items = []
    async for item in stream:
        items.append(item)

    assert items == ["hello", "world"]

@pytest.mark.asyncio
async def test_stream_map():
    """Test stream mapping"""
    stream = AsyncStream[int]()

    async def producer():
        for i in range(3):
            await stream.put(i)
        await stream.close()

    import asyncio
    asyncio.create_task(producer())

    mapped = stream.map(lambda x: x * 2)

    items = []
    async for item in mapped:
        items.append(item)

    assert items == [0, 2, 4]

@pytest.mark.asyncio
async def test_stream_filter():
    """Test stream filtering"""
    stream = AsyncStream[int]()

    async def producer():
        for i in range(5):
            await stream.put(i)
        await stream.close()

    import asyncio
    asyncio.create_task(producer())

    filtered = stream.filter(lambda x: x % 2 == 0)

    items = []
    async for item in filtered:
        items.append(item)

    assert items == [0, 2, 4]
