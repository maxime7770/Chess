from src import chess_main

import asyncio

async def main():
    chess_main.main()
    await asyncio.sleep(0)
    
asyncio.run(main())
