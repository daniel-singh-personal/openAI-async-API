# OpenAI Asynchronous API

This project demonstrates the use of the `OpenAIClient` class to interact with OpenAI's GPT models. The example script `example.py` shows how to use the asynchronous API by simply calling the make_async_requests method using the standard openAI prompt format. This is very useful for large scale jobs. requests_per_minute can be tuned based on your account limits in OpenAI.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Example](#example)

## Features
- Asynchronous requests to the OpenAI API to efficiently handle multiple inputs.
- Easy configuration of model parameters.

## Installation

To get started, clone the repository and install the necessary dependencies:

``` 
git clone https://github.com/https://github.com/daniel-singh-personal/openAI-async-API.git
cd OPENAI-ASYNC-API
pip install -r requirements.txt
```

## Usage

**Set your OpenAI API key**: Export your API key as an environment variable before running the script.

```
export OPENAI_API_KEY='your_openai_api_key_here'
```

## Configuration

The `OpenAIClient` class allows for easy configuration of various parameters:

- **`api_key`**: Your OpenAI API key.
- **`model`**: The model to be used. In this example, it is set to gpt-3.5-turbo-0125.
- **`temperature`**: Controls the randomness of the model's output. A lower temperature results in more deterministic responses.
- **`top_p`**: Controls the diversity via nucleus sampling, where 1 means taking the full distribution (default setting).
- **`frequency_penalty`**: Reduces the likelihood of repeated words or phrases in the output. Values range from 0 to 2.
- **`presence_penalty`**: Encourages the model to discuss new topics. Values range from 0 to 2.
- **`max_tokens`**: The maximum number of tokens (words or parts of words) to generate in the completion. Here it is set to 250.
- **`requests_per_minute`**: The rate limit for API requests, controlling how many requests can be made per minute.

## Example

These parameters are configured in the `OpenAIClient` instance in the `example.py` script:

```python
chat_client = OpenAIClient(
    api_key=os.environ["OPENAI_API_KEY"],
    model="gpt-3.5-turbo-0125",
    temperature=0,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    max_tokens=250,
    requests_per_minute=1,
)
```