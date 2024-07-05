import asyncio
import logging
from typing import Optional

import openai

from src.async_limiter import LimitedClientSession
from src.openai.cost_estimator import CostEstimator

logger = logging.getLogger(__name__)


class OpenAIClient:
    def __init__(
        self,
        api_key: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        requests_per_minute: Optional[int] = None,
        _ignore_cost_guardrail: Optional[bool] = False,
    ) -> None:
        """
        Initializes a wrapper for the OpenAI API.

        Parameters
        ----------
        api_key: str
            Your API key for accessing the OpenAI API. Generate your personal API key from
            https://platform.openai.com/account/api-keys and store it securely.
        model: str, optional
            The model to use for the task. See available models at https://platform.openai.com/docs/models
        temperature: float, optional
            Controls the randomness of the output. Lower values are suitable for standard NLP tasks,
            while higher values are better for more creative outputs (e.g., writing an article).
        max_tokens: int, optional
            Controls the maximum length of the output completion.
        top_p: float, optional
            Controls diversity via nucleus sampling. Achieves similar results as 'temperature',
            so you typically only need to tune one of these two parameters.
        frequency_penalty: float, optional
            Higher values decrease the likelihood of the model repeating the same words or sentences.
        presence_penalty: float, optional
            Higher values increase the likelihood of the model introducing new topics.
        requests_per_minute: int, optional
            The number of requests per minute to send to the API. Defaults to 5.
        """
        openai.api_key = api_key
        self.temperature = temperature or 0
        self.max_tokens = max_tokens or 256
        self.top_p = top_p or 1
        self.frequency_penalty = frequency_penalty or 0
        self.presence_penalty = presence_penalty or 0
        self.model = model or "gpt-3.5-turbo"
        self.requests_per_minute = requests_per_minute or 5

        self._ignore_cost_guardrail = _ignore_cost_guardrail

    def make_async_requests(self, user_inputs: list[list[dict[str, str]]]) -> list[str]:
        """
        Send a list of openai chat requests asynchronously.
        """
        for user_input in user_inputs:
            self._validate_user_input(user_input=user_input)
        if self._ignore_cost_guardrail:
            logger.warning("Ignoring cost guardrail, proceeding to send requests.")
        else:
            self._confirm_and_estimate_cost(prompts=[user_input])
        outputs = asyncio.run(self._send_all_requests(user_inputs=user_inputs))

        return outputs

    def _validate_user_input(self, user_input: list[dict[str, str]]) -> None:
        """
        Validates user input for the chat completion API. Raises an error if the input is not
        valid.

        Parameters
        ----------
        user_input: list[dict[str, str]]
            A list of dictionaries containing input for the chat completion endpoint. Each dictionary
            represents a chat input.
            See https://platform.openai.com/docs/guides/gpt/chat-completions-api for detailed
            documentation on the input requirements.
        """
        if not isinstance(user_input, list):
            raise ValueError("Invalid input. Input must be a list of dictionaries.")
        for entry in user_input:
            if "role" not in entry or "content" not in entry:
                raise ValueError("Incorrect API key.")
            if entry["role"] not in ("user", "system", "assistant"):
                raise ValueError("Incorrect API input role in input.")

    def _confirm_and_estimate_cost(self, prompts: list[list[dict[str, str]]]) -> None:
        """
        Takes a list of GPT prompts, estimates the upper and lower bound costs,
        prints them to the standard output, and awaits confirmation to continue
        from standard input. Raises a ValueError if no confirmation is received.

        Parameters
        ----------
        prompts: list[list[dict[str, str]]]
            A list of GPT prompts where each sublist represents a set of prompts.

        Returns
        -------
        None
        """
        prompt_texts = ["\n".join([message["content"] for message in prompt]) for prompt in prompts]

        cost_estimator = CostEstimator(model=self.model, max_tokens=self.max_tokens)
        lower_bound_cost, upper_bound_cost = cost_estimator.estimate_cost(text_list=prompt_texts)

        confirmation = input(
            f"The estimated cost of sending the provided data to the {self.model} model will be "
            f"between ${'{:.2f}'.format(lower_bound_cost)} - ${'{:.2f}'.format(upper_bound_cost)}"
            f" (USD). Would you like to proceed? (y/n)"
        )

        if confirmation.lower() == "y":
            logger.info("Proceeding to send requests.")
        else:
            raise ValueError("Requests cancelled.")

    async def _send_all_requests(self, user_inputs: list[list[dict[str, str]]]):
        """
        Sends all requests asynchronously.

        Parameters
        ----------
        user_inputs: list[list[dict[str, str]]]
            A list of user inputs where each sublist represents a set of inputs.

        Returns
        -------
        list: A list of responses received for each request.
        """
        tasks = []
        requests_per_second = self.requests_per_minute / 60
        async with LimitedClientSession(requests_per_second=requests_per_second) as session:
            openai.aiosession.set(session)
            for inputs in user_inputs:
                tasks.append(self._send_request(user_input=inputs))
            all_responses = await asyncio.gather(*tasks)
            await openai.aiosession.get().close()

        return all_responses

    async def _send_request(self, user_input: list[dict[str, str]]) -> str:
        """
        Sends a request to the OpenAI API for chat completion.

        Parameters
        ----------
        user_input: list[dict[str, str]]
            A list of dictionaries representing the user input.

        Returns
        -------
        str: The response received from the API.
        """
        attempt = 1
        while attempt <= 2:
            try:
                response = await openai.ChatCompletion.acreate(  # Note the 'a' in 'acreate'
                    model=self.model,
                    messages=user_input,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    frequency_penalty=self.frequency_penalty,
                    presence_penalty=self.presence_penalty,
                )
                return response["choices"][0]["message"]["content"]

            except Exception as e:
                logger.warning(f"API failed with error message '{e}'")
                if attempt == 1:
                    logger.info("Retrying...")
                attempt += 1

        return "API failed"
