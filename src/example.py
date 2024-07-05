from src.openai.openai_client import OpenAIClient
import os

# This is an example of how to use the OpenAIClient class.
chat_client = OpenAIClient(
                api_key=os.environ["OPENAI_API_KEY"],
                model="gpt-3.5-turbo-0125",
                temperature=0,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                max_tokens=250,
                requests_per_minute= 1,
            )

# Each prompt is list[list[dict[str, str]]]
example_input_prompts =  [
    [
        {
        "role": "system",
        "content": "You will be provided with a tweet, and your task is to classify its sentiment as positive, neutral, or negative."
        },
        {
        "role": "user",
        "content": "I loved the new Batman movie!"
        }
    ],
        [
        {
        "role": "system",
        "content": "You will be provided with a tweet, and your task is to classify its sentiment as positive, neutral, or negative."
        },
        {
        "role": "user",
        "content": "I hated the new Joker movie!"
        }
    ]
    ]

# Order of inputs is maintained
example_outputs = chat_client.make_async_requests(example_input_prompts)
print(example_outputs)