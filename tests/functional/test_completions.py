# SPDX-License-Identifier: Apache-2.0

# pylint: disable=redefined-outer-name, unused-argument

# Standard
from unittest.mock import AsyncMock, patch

# Third Party
from llama_stack.apis.inference import CompletionResponse
import pytest


@pytest.fixture
def mock_completion_response():
    with patch(
        "llama_stack.providers.remote.inference.vllm.vllm.process_completion_response"
    ) as mock_response:
        mock_response.return_value = CompletionResponse(
            stop_reason="end_of_turn",
            content="mock_response",
        )
        yield mock_response


@pytest.fixture
def mock_openai_completion(mock_completion_response):
    with patch(
        "openai.resources.completions.AsyncCompletions.create", new_callable=AsyncMock
    ) as mock_completion:
        yield mock_completion


def test_batch_prompts(client, mock_model_id, mock_openai_completion):
    args = []
    kwargs = {
        "model": mock_model_id,
        "prompt": ["test1", "test2"],
        "max_tokens": 1,
        "n": 3,
    }
    response = client.completions.create(*args, **kwargs)
    mock_openai_completion.assert_called()
    call_args = mock_openai_completion.call_args_list
    assert call_args[0].kwargs["model"] == mock_model_id
    assert call_args[0].kwargs["max_tokens"] == 1
    # 2 prompts * n of 3 == 6 expected completions
    assert len(call_args) == 6
    # ensure we get expected indices for our choices
    for i, choice in enumerate(response.choices):
        assert choice.index == i


def test_completion_ids(client, mock_model_id, mock_openai_completion):
    kwargs = {
        "model": mock_model_id,
        "prompt": ["test1", "test2"],
        "max_tokens": 1,
        "n": 3,
    }
    response1 = client.completions.create(**kwargs)
    response2 = client.completions.create(**kwargs)
    assert response1.id != response2.id


@pytest.mark.xfail(
    reason="bug in remote::vllm provider: applies Llama ChatFormat to completions",
    strict=True,
)
def test_completion_prompt_unchanged(client, mock_model_id, mock_openai_completion):
    prompt = "expected prompt"
    args = []
    kwargs = {
        "model": mock_model_id,
        "prompt": prompt,
    }
    client.completions.create(*args, **kwargs)
    mock_openai_completion.assert_called()
    call_kwargs = mock_openai_completion.call_args.kwargs
    assert call_kwargs["model"] == mock_model_id
    assert call_kwargs["prompt"] == prompt


def test_guided_decoding_valid_json_response(
    client, mock_model_id, mock_completion_response
):
    args = []
    kwargs = {
        "model": mock_model_id,
        "prompt": "foo bar",
        "extra_body": {"guided_choice": ["joy", "sadness"]},
    }
    with patch(
        "openai.resources.completions.AsyncCompletions.create", new_callable=AsyncMock
    ) as mock_openai_completion:
        mock_completion_response.return_value = CompletionResponse(
            stop_reason="end_of_turn",
            content="joy",
        )
        response = client.completions.create(*args, **kwargs)
        mock_openai_completion.assert_called()
        call_kwargs = mock_openai_completion.call_args.kwargs
        assert call_kwargs["extra_body"]
        assert response.choices
        assert response.choices[0].text == "joy"


@pytest.mark.xfail(
    reason="figure out how to round-trip guided choice directly instead of via json schema",
    strict=True,
)
def test_guided_decoding_uses_guided_choice(
    client, mock_model_id, mock_completion_response
):
    guided_choices = ["joy", "sadness"]
    args = []
    kwargs = {
        "model": mock_model_id,
        "prompt": "foo bar",
        "extra_body": {"guided_choice": guided_choices},
    }
    with patch(
        "openai.resources.completions.AsyncCompletions.create", new_callable=AsyncMock
    ) as mock_openai_completion:
        mock_completion_response.return_value = CompletionResponse(
            stop_reason="end_of_turn",
            content="joy",
        )
        response = client.completions.create(*args, **kwargs)
        mock_openai_completion.assert_called()
        call_kwargs = mock_openai_completion.call_args.kwargs
        assert call_kwargs["extra_body"]
        assert call_kwargs["extra_body"]["guided_choice"]
        assert call_kwargs["extra_body"]["guided_choice"] == guided_choices
        assert response.choices
        assert response.choices[0].text == "joy"


def test_guided_decoding_invalid_json_response(
    client, mock_model_id, mock_completion_response
):
    args = []
    kwargs = {
        "model": mock_model_id,
        "prompt": "foo bar",
        "extra_body": {"guided_choice": ["joy", "sadness"]},
    }
    with patch(
        "openai.resources.completions.AsyncCompletions.create", new_callable=AsyncMock
    ) as mock_openai_completion:
        mock_completion_response.return_value = CompletionResponse(
            stop_reason="end_of_turn",
            content="something that is not valid json",
        )
        response = client.completions.create(*args, **kwargs)
        mock_openai_completion.assert_called()
        call_kwargs = mock_openai_completion.call_args.kwargs
        assert call_kwargs["extra_body"]
        assert response.choices
        assert response.choices[0].text == "something that is not valid json"


def test_temperature_0(client, mock_model_id, mock_openai_completion):
    kwargs = {
        "model": mock_model_id,
        "prompt": "foo bar",
        "temperature": 0,
    }
    client.completions.create(**kwargs)
    mock_openai_completion.assert_called()
    call_kwargs = mock_openai_completion.call_args.kwargs
    assert call_kwargs["temperature"] == 0
