import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Thread

""" 
“await” is not a function
await is a keyword that can be used only inside a function declared with async def.
When the interpreter hits an await expression it:

Evaluates the right-hand side, which must be an awaitable (a coroutine object, a Task, or any object implementing __await__).

Suspends the current coroutine, handing control back to the event-loop so something else can run.

Resumes the coroutine when the awaitable is finished, passing its result back to the left-shand side.
"""

OPTION = 'B'

async def task(name, delay):
    print(f"Start {name}")
    # await 
    await asyncio.sleep(delay)
    print(f"Done {name}")

if OPTION == 'A':
    # 1. asyncio.gather
    async def main():
        await asyncio.gather(
            task("A", 2),
            task("B", 1),
            task("C", 3),
        )

    asyncio.run(main())

elif OPTION == 'B':
    # 2. asyncio.create_task
    async def main():
        t1 = asyncio.create_task(task("X", 3))
        t2 = asyncio.create_task(task("Y", 1))
        
        print("Waiting for tasks...")
        await t2
        await t1

    asyncio.run(main())