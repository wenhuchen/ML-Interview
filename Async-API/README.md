# OpenAI API Concurrency Comparison

This project demonstrates and compares different approaches to making concurrent API calls to OpenAI's ChatGPT API, measuring the performance differences between threading, sequential execution, and asyncio.

## Overview

The `run.py` script contains three different implementations for making 10 API calls to OpenAI:

1. **ThreadPoolExecutor** - Uses thread-based concurrency
2. **Sequential execution** - Makes API calls one after another
3. **Asyncio with `asyncio.to_thread()`** - Uses async/await with thread delegation

## Requirements

- Python 3.7+
- OpenAI Python SDK
- Valid OpenAI API key set as `OPENAI_API_KEY` environment variable

## Installation

```bash
pip install openai
```

## Setup

Set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

Simply run the script:

```bash
python run.py
```

The script will execute all three approaches sequentially and display:
- The responses from each API call
- Execution time for each approach
- Performance comparison between the methods

## Code Structure

### `prompt_chatgpt()` Function

A utility function that:
- Sends a prompt to OpenAI's ChatGPT API
- Supports both legacy and new OpenAI SDK versions
- Configurable parameters: model, temperature, max_tokens, timeout
- Returns the assistant's response content

### Three Execution Patterns

1. **Threading approach**: Uses `ThreadPoolExecutor` to submit 10 concurrent tasks
2. **Sequential approach**: Makes 10 API calls in a simple for loop
3. **Asyncio approach**: Uses `asyncio.gather()` with `asyncio.to_thread()` for concurrent execution

## Expected Output

The script will show timing comparisons, typically demonstrating that:
- Threading and asyncio approaches are significantly faster than sequential execution
- All approaches produce the same functional results
- Actual performance may vary based on network latency and API response times

## Notes

- The script is compatible with both older (v0.x) and newer (v1.x) versions of the OpenAI Python SDK
- API key is read from environment variables for security
- Each approach sends the same 10 prompts for fair performance comparison