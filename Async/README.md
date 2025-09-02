# Async Programming in Python

This directory contains examples and educational materials about asynchronous programming in Python using `asyncio`. Async programming is crucial for building high-performance applications that need to handle I/O operations efficiently.

## Table of Contents

- [What is Async Programming?](#what-is-async-programming)
- [Key Concepts](#key-concepts)
- [Scripts Overview](#scripts-overview)
- [Running the Examples](#running-the-examples)
- [Best Practices](#best-practices)
- [Common Pitfalls](#common-pitfalls)

## What is Async Programming?

Asynchronous programming allows your program to handle multiple operations concurrently without blocking the execution thread. Instead of waiting for one operation to complete before starting another, async programs can start multiple operations and handle them as they complete.

### Synchronous vs Asynchronous

**Synchronous (Blocking):**
```python
def sync_example():
    result1 = slow_operation_1()  # Waits 3 seconds
    result2 = slow_operation_2()  # Waits 2 seconds
    return result1, result2
# Total time: 5 seconds
```

**Asynchronous (Non-blocking):**
```python
async def async_example():
    result1, result2 = await asyncio.gather(
        slow_operation_1(),  # Starts immediately
        slow_operation_2()   # Starts immediately
    )
    return result1, result2
# Total time: 3 seconds (max of both operations)
```

## Key Concepts

### 1. Coroutines
Functions defined with `async def` that can be paused and resumed:
```python
async def my_coroutine():
    await asyncio.sleep(1)  # Pauses here
    return "Done"
```

### 2. Await
The `await` keyword pauses execution until the awaited operation completes:
```python
result = await some_async_function()
```

### 3. Event Loop
The core of async programming - manages and executes coroutines:
```python
asyncio.run(main())  # Creates and runs event loop
```

### 4. Tasks
Ways to schedule coroutines for concurrent execution:
```python
task1 = asyncio.create_task(coroutine1())
task2 = asyncio.create_task(coroutine2())
await asyncio.gather(task1, task2)
```

## Scripts Overview

### 1. `async_basic.py` - Basic Async Concepts

This script demonstrates fundamental async patterns with two different approaches:

**Option A: `asyncio.gather()`**
- Runs multiple coroutines concurrently
- Waits for all to complete
- Returns results in the same order as input

**Option B: `asyncio.create_task()`**
- Creates tasks that run concurrently
- Allows more control over individual task execution
- Can await tasks individually

**Key Learning Points:**
- How `await` suspends execution
- Difference between `gather()` and `create_task()`
- Task scheduling and execution order

### 2. `async_api_call.py` - Real-world API Example

Demonstrates async programming with actual API calls using Google's Gemini API:

**Features:**
- Converts synchronous API calls to async using `asyncio.to_thread()`
- Makes multiple API calls concurrently
- Shows practical async usage for I/O-bound operations

**Key Learning Points:**
- Converting sync functions to async
- Using `asyncio.to_thread()` for blocking operations
- Real-world async API usage patterns

### 3. `async_multi_thread.py` - Async vs Threading Comparison

Compares async programming with traditional threading:

**Features:**
- Demonstrates async tasks within threads
- Shows performance comparison between single-threaded and multi-threaded execution
- Uses `ThreadPoolExecutor` for comparison

**Key Learning Points:**
- When to use async vs threading
- Performance implications of different concurrency models
- Event loop management in different contexts

## Running the Examples

### Prerequisites
```bash
pip install google-genai  # For async_api_call.py
```

### Basic Example
```bash
python async_basic.py
```

### API Example (requires API key)
```bash
# Edit async_api_call.py and add your API key
python async_api_call.py
```

### Multi-threading Example
```bash
python async_multi_thread.py
```

## Best Practices

### 1. Use `asyncio.gather()` for Multiple Operations
```python
# Good
results = await asyncio.gather(
    fetch_data_1(),
    fetch_data_2(),
    fetch_data_3()
)

# Avoid
result1 = await fetch_data_1()
result2 = await fetch_data_2()
result3 = await fetch_data_3()
```

### 2. Handle Exceptions Properly
```python
async def safe_operation():
    try:
        result = await risky_operation()
        return result
    except Exception as e:
        print(f"Operation failed: {e}")
        return None
```

### 3. Use `asyncio.to_thread()` for CPU-bound Operations
```python
# For blocking I/O or CPU-intensive tasks
result = await asyncio.to_thread(cpu_intensive_function, arg1, arg2)
```

### 4. Proper Resource Management
```python
async def with_context_manager():
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()
```

## Common Pitfalls

### 1. Forgetting `await`
```python
# Wrong - returns a coroutine object, not the result
result = async_function()

# Correct
result = await async_function()
```

### 2. Blocking the Event Loop
```python
# Wrong - blocks the entire event loop
async def bad_example():
    time.sleep(5)  # Blocking!

# Correct
async def good_example():
    await asyncio.sleep(5)  # Non-blocking
```

### 3. Mixing Sync and Async Code Incorrectly
```python
# Wrong - can't call async from sync context
def sync_function():
    result = await async_function()  # Error!

# Correct
async def async_wrapper():
    result = await async_function()
    return result

# Or use asyncio.run()
def sync_function():
    result = asyncio.run(async_function())
    return result
```

### 4. Not Handling Task Cancellation
```python
# Good - handle cancellation gracefully
async def cancellable_task():
    try:
        await long_running_operation()
    except asyncio.CancelledError:
        print("Task was cancelled")
        raise  # Re-raise to properly handle cancellation
```

## When to Use Async

**Use Async For:**
- I/O-bound operations (network requests, file operations, database queries)
- Web servers and APIs
- Real-time applications
- Scraping and data collection

**Don't Use Async For:**
- CPU-bound operations (use multiprocessing instead)
- Simple scripts with no I/O
- Operations that must run sequentially

## Further Reading

- [Python asyncio Documentation](https://docs.python.org/3/library/asyncio.html)
- [Real Python Async Tutorial](https://realpython.com/async-io-python/)
- [AsyncIO Best Practices](https://docs.python.org/3/library/asyncio-dev.html)

## Performance Tips

1. **Batch Operations**: Group related async operations together
2. **Connection Pooling**: Reuse connections when possible
3. **Limit Concurrency**: Use semaphores to prevent overwhelming systems
4. **Monitor Resources**: Keep track of memory and connection usage

```python
# Example with concurrency limiting
async def limited_concurrent_operations():
    semaphore = asyncio.Semaphore(10)  # Max 10 concurrent operations
    
    async def limited_operation():
        async with semaphore:
            return await some_operation()
    
    tasks = [limited_operation() for _ in range(100)]
    results = await asyncio.gather(*tasks)
    return results
```

This concludes our async programming guide. Experiment with the examples and try building your own async applications!
