import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

async def task(delay):
    loop = asyncio.get_running_loop()
    task = loop.create_task(asyncio.sleep(delay))
    await task
    return 42

async def tasks(times):
    for step in range(1, times + 1):
        await task(step)

def main(name: int):
    # Schedule nested() to run soon concurrently
    # with "main()".
    # task = asyncio.create_task(nested())
    time_start = time.time()
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    loop.run_until_complete(tasks(3))
    time_end = time.time()
    print(f"Single Task {name} Time taken: {time_end - time_start} seconds")

if __name__ == "__main__":
    futures = []
    time_start = time.time()

    OPTION = 'multi-thread'

    if OPTION == 'multi-thread':
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures.append(executor.submit(main, 1))
            futures.append(executor.submit(main, 2))
            futures.append(executor.submit(main, 3))
            futures.append(executor.submit(main, 4))

        for future in futures:
            future.result()
    else:
        main(1)
        main(2)
        main(3)
        main(4)

    time_end = time.time()
    print(f"Total Time taken: {time_end - time_start} seconds")
    print("All tasks completed")