# Standard
import json
import time

# Third Party
from llama_stack_client import LlamaStackClient
from llama_stack_client.types.shared_params.response_format import (
    JsonSchemaResponseFormat,
)
from llama_stack_client.types.shared_params.sampling_params import (
    SamplingParams,
    StrategyTopPSamplingStrategy,
)
from openai.types.completion import Completion as OpenAICompletion
from openai.types.completion_choice import CompletionChoice as OpenAICompletionChoice
import httpx


class Completions:
    def __init__(self, llama_stack_client):
        self.lls_client = llama_stack_client

    def create(self, *args, **kwargs):
        model_id = kwargs["model"]
        prompts = kwargs["prompt"]

        # TODO: This is a pretty hacky way to do batch completions -
        # basically just de-batches them...
        if not isinstance(prompts, list):
            prompts = [prompts]

        sampling_params = SamplingParams()
        n = kwargs.get("n", 1)

        max_tokens = kwargs.get("max_tokens", None)
        if max_tokens:
            sampling_params["max_tokens"] = max_tokens

        temperature = kwargs.get("temperature", None)
        if temperature:
            # TODO: hardcoded top_p of 1.0 ...
            top_p_sampling_strategy = StrategyTopPSamplingStrategy(
                type="top_p",
                temperature=temperature,
                top_p=1.0,
            )
            sampling_params["strategy"] = top_p_sampling_strategy

        # TODO: make a separate function, handle other options besides
        # just guided_choice
        response_format = None
        extra_body = kwargs.get("extra_body", {})
        guided_choice = extra_body.get("guided_choice", [])
        if guided_choice:
            pattern_choices = "|".join(guided_choice)
            schema = {
                "type": "array",
                "minItems": 1,
                "maxItems": 1,
                "items": {"type": "string", "pattern": f"^({pattern_choices})$"},
            }
            response_format = JsonSchemaResponseFormat(
                type="json_schema",
                json_schema=schema,
            )

        # TODO: pull into separate function, ensure all stop reasons
        # are handled - this dict may not cover all cases
        stop_reason_map = {
            "end_of_turn": "stop",
            "end_of_message": "stop",
            "out_of_tokens": "length",
        }
        choices = []
        # "n" is the number of completions to generate per prompt
        for i in range(0, n):
            # and we may have multiple prompts, if batching was used

            # TODO: see if this can get wired up to LlamaStack's batch
            # inference API
            for prompt in prompts:
                lls_result = self.lls_client.inference.completion(
                    model_id=model_id,
                    content=prompt,
                    sampling_params=sampling_params,
                    response_format=response_format,
                )

                if guided_choice:
                    # TODO: error handling - this is very naive
                    # parsing, assuming we always get well-formed JSON
                    # response with at least one element
                    text = json.loads(lls_result.content)[0]
                else:
                    text = lls_result.content

                choice = OpenAICompletionChoice(
                    # TODO: "i" is the wrong index, but doesn't seem
                    # to matter right now so fix later to account for
                    # the fact we have 2 loops here
                    index=i,
                    text=text,
                    finish_reason=stop_reason_map[lls_result.stop_reason],
                )
                choices.append(choice)
        # TODO: a real id value, or maybe just a uuid?
        return OpenAICompletion(
            id="foo",
            choices=choices,
            created=int(time.time()),
            model=model_id,
            object="text_completion",
        )


class ChatCompletions:
    def __init__(self, llama_stack_client):
        self.lls_client = llama_stack_client

    def create(self, *args, **kwargs):
        # TODO: This obviously needs to get filled out and logic
        # deduplicated with the regular completions where possible
        print(f"!!! calling completions.create with {args} and {kwargs}")
        return OpenAICompletion(
            id="foo",
            choices=[],
            created=0,
            model="foo",
            object="text_completion",
        )


class Chat:
    completions: ChatCompletions

    def __init__(self, llama_stack_client):
        self.lls_client = llama_stack_client
        self.completions = ChatCompletions(self.lls_client)


class Models:
    def __init__(self, llama_stack_client):
        self.lls_client = llama_stack_client

    def list(self, *args, **kwargs):
        # TODO: this needs to convert response values from Llama Stack
        # to OpenAI format
        return self.lls_client.models.list()


class OpenAIClientAdapter:
    completions: Completions
    chat: Chat

    def __init__(self, llama_stack_client):
        self.lls_client = llama_stack_client
        if not self.lls_client:
            raise ValueError("A `llama_stack_client` must be provided.")

        self.completions = Completions(self.lls_client)
        self.chat = Chat(self.lls_client)
        self.models = Models(self.lls_client)

    @property
    def base_url(self) -> httpx.URL:
        return self.lls_client.base_url

    def get(self, *args, **kwargs):
        return self.lls_client.get(*args, **kwargs)
