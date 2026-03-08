import asyncio
import websockets

async def listen():
    async with websockets.connect("ws://localhost:8000/ws") as ws:
        print("Listening for events... (sign a letter, then rest)")
        while True:
            print(await ws.recv())

asyncio.run(listen())
