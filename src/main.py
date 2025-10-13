from assistant.Assistant import Assistant
import asyncio

async def main():
    """
    The main function is the entry point for the assistant.
    It is responsible for running the assistant.
    """
    assistant = Assistant()
    await assistant.initialize()
    try:  
        await assistant.run()
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\nKeyboard interrupt received. Shutting down assistant...")
        await assistant.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
