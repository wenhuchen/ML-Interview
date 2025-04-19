from google import genai
from google.genai import types
import asyncio

""" 
“await” is not a function;
We need to convert a sync function to an async thread function.
"""

client = genai.Client(
    api_key="YOUR API KEY",
)

def generate_sync(input: str):
    model = 'gemini-1.5-flash'
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=input),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        response_mime_type="text/plain",
    )
    print('Start generating...')
    content = client.models.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config,
    )
    content = content.candidates[0].content.parts[0].text
    return content

async def generate(input: str):
    return await asyncio.to_thread(generate_sync, input)

async def main():
    results = await asyncio.gather(
        generate("What are the best places to visit in the world?"),
        generate("What are the best places to visit in China?"),
    )
    print(results)

if __name__ == "__main__":
    asyncio.run(main())