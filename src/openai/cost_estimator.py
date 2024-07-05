import tiktoken
import logging

logger = logging.getLogger(__name__)


class CostEstimator:
    """
    Estimates the cost of using the OpenAI API.

    Costs are calculated in USD and are based on both input prompt and output response lengths. 
    The class defines a dictionary with the model name as the key and another dictionary as the value,
    {'input': input_rate, 'output': output_rate}, where the rates are in USD per 1K tokens.
    For OpenAI pricing details, visit:
    https://openai.com/pricing

    """

    model_cost_dict = {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-32k": {"input": 0.06, "output": 0.12},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        "gpt-3.5-turbo-0125": {"input": 0.0005, "output": 0.0015},
        "text-embedding-ada-002": {"input": 0.0001},
    }
    model_to_task_dict = {
        "gpt-4": "Chat",
        "gpt-4-32k": "Chat",
        "gpt-3.5-turbo": "Chat",
        "gpt-3.5-turbo-0125": "Chat",
        "text-embedding-ada-002": "Embedding",
    }

    def __init__(self, model, max_tokens: int = None) -> None:
        """
        Initializes CostEstimator with a specified model and maximum tokens for output.

        Parameters
        ----------
        model : str
            The OpenAI model to be used.
        max_tokens : int, optional
            The maximum length of the output in tokens. Defaults to None.
        """

        if model not in self.model_cost_dict:
            raise ValueError(f"Unsupported model.")
        self.model = model
        self.max_tokens = max_tokens or 256

        model_type = self.model_to_task_dict[self.model].upper()
        user_input = input(
            f"Estimate costs for the {self.model} model. (y/n)?"
        )
        if user_input.lower() == "y":
            logger.info("Estimating costs...")
        else:
            logger.info("Skipping cost estimates and running...")

    def estimate_cost(self, text_list: list[str]) -> tuple[float, float]:
        """
        Estimates the cumulative upper and lower bound costs for a list of texts to be sent to the API.

        Parameters
        ----------
        text_list : list[str]
            List of texts to be sent to the API.

        Returns
        -------
        tuple[float, float]
            A tuple with the lower and upper bounds of the cost for the whole batch.
        """

        batch_costs = [self._calculate_single_cost(text) for text in text_list]
        lower_bound_sum = sum([item[0] for item in batch_costs])
        upper_bound_sum = sum([item[1] for item in batch_costs])

        return lower_bound_sum, upper_bound_sum

    def _calculate_single_cost(self, text: str) -> tuple[float, float]:
        """
        Calculates the lower and upper bounds of the cost for calling the API with the given text.

        Parameters
        ----------
        text : str
            The text to be sent to the API.

        Returns
        -------
        tuple[float, float]
            A tuple with the lower and upper bounds of the cost.
        """
        encoding = tiktoken.encoding_for_model(self.model)

        token_list = encoding.encode(text)
        n_tokens_input = len(token_list)

        n_tokens_one_k = n_tokens_input / 1000

        if self.model != "text-embedding-ada-002":  # text-ada is for embeddings, so no output here
            output_tokens_max_one_k = self.max_tokens / 1000
            output_cost_max = output_tokens_max_one_k * self.model_cost_dict[self.model]["output"]
        else:
            output_cost_max = 0

        lower_bound_cost = n_tokens_one_k * self.model_cost_dict[self.model]["input"]
        upper_bound_cost = lower_bound_cost + output_cost_max

        return lower_bound_cost, upper_bound_cost
