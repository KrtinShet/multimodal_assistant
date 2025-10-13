from typing import TypeVar, Generic, Optional, Callable, AsyncIterator
import asyncio

T = TypeVar('T')
U = TypeVar('U')

class AsyncStream(Generic[T]):
    """Generic async stream implementation"""

    def __init__(self, source: AsyncIterator[T] = None):
        self._source = source
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=10)
        self._closed = False

    async def next(self) -> Optional[T]:
        """Get next item from stream, waiting until data or closure."""

        while True:
            if self._closed and self._queue.empty():
                return None

            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=0.5)
                return item
            except asyncio.TimeoutError:
                if self._closed and self._queue.empty():
                    return None
                continue

    async def put(self, item: T):
        """Add item to stream"""
        if not self._closed:
            await self._queue.put(item)

    async def close(self):
        """Close stream"""
        self._closed = True

    def map(self, transform: Callable[[T], U]) -> 'AsyncStream[U]':
        """Transform stream elements"""
        output_stream = AsyncStream[U]()

        async def _transform():
            async for item in self:
                transformed = transform(item)
                await output_stream.put(transformed)
            await output_stream.close()

        asyncio.create_task(_transform())
        return output_stream

    def filter(self, predicate: Callable[[T], bool]) -> 'AsyncStream[T]':
        """Filter stream elements"""
        output_stream = AsyncStream[T]()

        async def _filter():
            async for item in self:
                if predicate(item):
                    await output_stream.put(item)
            await output_stream.close()

        asyncio.create_task(_filter())
        return output_stream

    async def __aiter__(self):
        """Async iteration support"""
        while True:
            item = await self.next()
            if item is None:
                break
            yield item
