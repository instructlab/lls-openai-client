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
from openai.types.chat.chat_completion import ChatCompletion as OpenAIChatCompletion
from openai.types.chat.chat_completion import Choice as OpenAIChatCompletionChoice
from openai.types.chat.chat_completion_message import (
    ChatCompletionMessage as OpenAIChatCompletionMessage,
)
from openai.types.completion import Completion as OpenAICompletion
from openai.types.completion_choice import CompletionChoice as OpenAICompletionChoice
import httpx

_STOP_REASON_MAP = {
    "end_of_turn": "stop",
    "end_of_message": "stop",
    "out_of_tokens": "length",
}


def _map_stop_reason(lls_stop_reason: str) -> str:
    return _STOP_REASON_MAP.get(lls_stop_reason, "")


class Completions:
    def __init__(self, llama_stack_client):
        self.lls_client = llama_stack_client

    def create(self, *_args, **kwargs):
        model_id = kwargs.get("model", None)
        prompts = kwargs.get("prompt", None)

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

                text = lls_result.content
                if guided_choice:
                    try:
                        json_results = json.loads(lls_result.content)
                        if json_results and isinstance(json_results, list):
                            text = json_results[0]
                    except json.decoder.JSONDecodeError:
                        # invalid JSON, so just leave the text as the raw content
                        pass

                choice = OpenAICompletionChoice(
                    # TODO: "i" is the wrong index, but doesn't seem
                    # to matter right now so fix later to account for
                    # the fact we have 2 loops here
                    index=i,
                    text=text,
                    finish_reason=_map_stop_reason(lls_result.stop_reason),
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

    def create(self, *_args, **kwargs):
        model_id = kwargs.get("model", None)
        messages = kwargs.get("messages", None)
        n = kwargs.get("n", 1)
        sampling_params = SamplingParams()
        response_format = None

        choices = []
        # "n" is the number of completions to generate per prompt
        for i in range(0, n):
            lls_result = self.lls_client.inference.chat_completion(
                model_id=model_id,
                messages=messages,
                sampling_params=sampling_params,
                response_format=response_format,
            )

            completion_message = lls_result.completion_message
            message = OpenAIChatCompletionMessage(
                role=completion_message.role,
                content=completion_message.content or "",
                tool_calls=completion_message.tool_calls,
            )

            choice = OpenAIChatCompletionChoice(
                # TODO: "i" is the wrong index, but doesn't seem
                # to matter right now so fix later to account for
                # the fact we have 2 loops here
                index=i,
                message=message,
                finish_reason=_map_stop_reason(completion_message.stop_reason),
            )
            choices.append(choice)

        return OpenAIChatCompletion(
            id="foo",
            choices=choices,
            created=0,
            model="foo",
            object="chat.completion",
        )


class Chat:
    completions: ChatCompletions

    def __init__(self, llama_stack_client):
        self.lls_client = llama_stack_client
        self.completions = ChatCompletions(self.lls_client)


class Models:
    def __init__(self, llama_stack_client):
        self.lls_client = llama_stack_client

    def list(self, *_args, **_kwargs):
        # TODO: this needs to convert response values from Llama Stack
        # to OpenAI format
        return self.lls_client.models.list()


class OpenAIClientAdapter:
    completions: Completions
    chat: Chat

    def __init__(self, llama_stack_client: LlamaStackClient):
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
