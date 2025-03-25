# SPDX-License-Identifier: Apache-2.0

# pylint: disable=redefined-outer-name, unused-argument

# Standard
from unittest.mock import MagicMock, patch
import json

# Third Party
import httpx
import pytest

# Local
from .conftest import MOCK_MODEL_ID


@pytest.fixture
def vllm_response():
    return {
        "model": MOCK_MODEL_ID,
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "mock response",
                    "tool_calls": [],
                },
            },
        ],
    }


@pytest.fixture(autouse=True)
def mock_httpx_send(vllm_response):
    with patch("httpx.AsyncClient.send") as mock_send:
        request = MagicMock()
        request.headers = {}
        mock_send.return_value = httpx.Response(
            status_code=200,
            json=vllm_response,
            request=request,
        )
        yield mock_send


# list scoped to the entire module so we can track ids seen across
# responses
@pytest.fixture(scope="module")
def id_tracker():
    return []


@pytest.mark.parametrize(
    "client_request,vllm_request",
    [
        (
            {
                "model": MOCK_MODEL_ID,
                "messages": [{"role": "user", "content": "user prompt"}],
                "n": 3,
            },
            {
                "model": MOCK_MODEL_ID,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "user prompt",
                            }
                        ],
                    }
                ],
                "max_tokens": 4096,
                "stream": False,
                "temperature": 1.0,
                "tools": None,
                "top_p": 1.0,
            },
        ),
        (
            {
                "model": MOCK_MODEL_ID,
                "messages": [
                    {
                        "role": "system",
                        "content": "system prompt",
                    },
                    {"role": "user", "content": "user prompt"},
                ],
            },
            {
                "model": MOCK_MODEL_ID,
                "messages": [
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": "system prompt",
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "user prompt",
                            }
                        ],
                    },
                ],
                "max_tokens": 4096,
                "stream": False,
                "temperature": 1.0,
                "tools": None,
                "top_p": 1.0,
            },
        ),
        (
            {
                "model": MOCK_MODEL_ID,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "user prompt"}],
                    }
                ],
            },
            {
                "model": MOCK_MODEL_ID,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "user prompt",
                            }
                        ],
                    }
                ],
                "max_tokens": 4096,
                "stream": False,
                "temperature": 1.0,
                "tools": None,
                "top_p": 1.0,
            },
        ),
        (
            {
                "model": MOCK_MODEL_ID,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "user prompt"}],
                    }
                ],
                "max_tokens": 4,
                "temperature": 0.5,
            },
            {
                "model": MOCK_MODEL_ID,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "user prompt",
                            }
                        ],
                    }
                ],
                "max_tokens": 4,
                "stream": False,
                "temperature": 0.5,
                "tools": None,
                "top_p": 1.0,
            },
        ),
        (
            {
                "model": MOCK_MODEL_ID,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "user prompt"}],
                    }
                ],
                "top_p": 0.8,
            },
            {
                "model": MOCK_MODEL_ID,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "user prompt",
                            }
                        ],
                    }
                ],
                "max_tokens": 4096,
                "stream": False,
                "temperature": 1.0,
                "tools": None,
                "top_p": 0.8,
            },
        ),
        (
            {
                "model": MOCK_MODEL_ID,
                "messages": [
                    {"role": "user", "content": "Call the get_price function"}
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_price",
                        },
                    }
                ],
            },
            {
                "model": MOCK_MODEL_ID,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Call the get_price function",
                            }
                        ],
                    }
                ],
                "max_tokens": 4096,
                "stream": False,
                "temperature": 1.0,
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_price",
                            "description": None,
                            "parameters": {
                                "properties": {},
                                "required": [],
                                "type": "object",
                            },
                        },
                    }
                ],
                "top_p": 1.0,
            },
        ),
        pytest.param(
            {
                "model": MOCK_MODEL_ID,
                "messages": [
                    {"role": "user", "content": "Call the get_price function"}
                ],
                "tool_choice": "auto",
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_price",
                        },
                    }
                ],
            },
            {
                "model": MOCK_MODEL_ID,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Call the get_price function",
                            }
                        ],
                    }
                ],
                "max_tokens": 4096,
                "stream": False,
                "temperature": 1.0,
                "tool_choice": "auto",
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_price",
                            "description": None,
                            "parameters": {
                                "properties": {},
                                "required": [],
                                "type": "object",
                            },
                        },
                    }
                ],
                "top_p": 1.0,
            },
            marks=pytest.mark.xfail(
                reason="bug in remote:vllm provider: does not pass tool_choice",
                strict=True,
            ),
        ),
        (
            {
                "model": MOCK_MODEL_ID,
                "messages": [{"role": "user", "content": "user prompt"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "foo_bar",
                            "description": "my user function",
                            "parameters": {
                                "properties": {
                                    "param1": {
                                        "type": "string",
                                        "description": "a string",
                                    },
                                },
                                "required": ["param1"],
                            },
                        },
                    }
                ],
            },
            {
                "model": MOCK_MODEL_ID,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "user prompt",
                            }
                        ],
                    }
                ],
                "max_tokens": 4096,
                "stream": False,
                "temperature": 1.0,
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "foo_bar",
                            "description": "my user function",
                            "parameters": {
                                "properties": {
                                    "param1": {
                                        "type": "string",
                                        "description": "a string",
                                    },
                                },
                                "required": ["param1"],
                                "type": "object",
                            },
                        },
                    }
                ],
                "top_p": 1.0,
            },
        ),
        pytest.param(
            {
                "model": MOCK_MODEL_ID,
                "messages": [
                    {"role": "user", "content": "user prompt"},
                    {
                        "role": "tool",
                        "tool_call_id": "call_abc123",
                        "content": "success",
                    },
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "foo_bar",
                            "description": "my user function",
                            "parameters": {
                                "properties": {
                                    "param1": {
                                        "type": "string",
                                        "description": "a string",
                                    },
                                },
                                "required": ["param1"],
                                "type": "object",
                            },
                        },
                    }
                ],
            },
            {
                "model": MOCK_MODEL_ID,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "user prompt",
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_abc123",
                        "content": [
                            {
                                "type": "text",
                                "text": "success",
                            }
                        ],
                    },
                ],
                "max_tokens": 4096,
                "stream": False,
                "temperature": 1.0,
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "foo_bar",
                            "description": "my user function",
                            "parameters": {
                                "properties": {
                                    "param1": {
                                        "type": "string",
                                        "description": "a string",
                                    },
                                },
                                "required": ["param1"],
                                "type": "object",
                            },
                        },
                    }
                ],
                "top_p": 1.0,
            },
            marks=pytest.mark.xfail(
                reason="bug in remote:vllm provider: does not pass tool_call_id",
                strict=True,
            ),
        ),
    ],
)
def test_chat_completion_requests_non_streaming(
    client,
    mock_httpx_send,
    client_request,
    vllm_request,
):
    num_expected_calls = client_request.get("n", 1)
    client.chat.completions.create(**client_request)

    # Verify we sent the expected number of requests to vLLM
    mock_httpx_send.assert_called()
    send_calls = mock_httpx_send.call_args_list
    assert len(send_calls) == num_expected_calls

    for send_call in send_calls:
        # Verify each request to vLLM matches the expected json
        request = send_call.args[0]
        assert isinstance(request, httpx.Request)
        request_json = json.loads(request.content)
        assert request_json == vllm_request


@pytest.mark.parametrize(
    "vllm_response,client_response",
    [
        (
            {
                "model": MOCK_MODEL_ID,
                "choices": [
                    {
                        "finish_reason": "stop",
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "mock response",
                            "tool_calls": [],
                        },
                    },
                    {
                        "finish_reason": "stop",
                        "index": 1,
                        "message": {
                            "role": "assistant",
                            "content": "mock response",
                            "tool_calls": [],
                        },
                    },
                ],
            },
            {
                "model": MOCK_MODEL_ID,
                "choices": [
                    {
                        "finish_reason": "stop",
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "mock response",
                            "tool_calls": [],
                        },
                    },
                    {
                        "finish_reason": "stop",
                        "index": 1,
                        "message": {
                            "role": "assistant",
                            "content": "mock response",
                            "tool_calls": [],
                        },
                    },
                ],
            },
        ),
        (
            {
                "model": MOCK_MODEL_ID,
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "What can I assist you with today?",
                            "tool_calls": [],
                        },
                    },
                ],
            },
            {
                "model": MOCK_MODEL_ID,
                "choices": [
                    {
                        "finish_reason": "stop",
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "What can I assist you with today?",
                            "tool_calls": [],
                        },
                    },
                ],
            },
        ),
        (
            {
                "model": MOCK_MODEL_ID,
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {
                            "role": "assistant",
                            "content": "You could provide more",
                            "tool_calls": [],
                        },
                    },
                ],
            },
            {
                "model": MOCK_MODEL_ID,
                "choices": [
                    {
                        "finish_reason": "length",
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "You could provide more",
                            "tool_calls": [],
                        },
                    },
                ],
            },
        ),
        (
            {
                "model": MOCK_MODEL_ID,
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_abc123",
                                    "type": "function",
                                    "function": {
                                        "name": "get_price",
                                        "arguments": "{}",
                                    },
                                },
                            ],
                        },
                    },
                ],
            },
            {
                "model": MOCK_MODEL_ID,
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_abc123",
                                    "type": "function",
                                    "function": {
                                        "name": "get_price",
                                        "arguments": "{}",
                                    },
                                }
                            ],
                        },
                    },
                ],
            },
        ),
        (
            {
                "model": MOCK_MODEL_ID,
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_abc123",
                                    "type": "function",
                                    "function": {
                                        "name": "foo_bar",
                                        "arguments": '{"param1":"baz"}',
                                    },
                                },
                            ],
                        },
                    },
                ],
            },
            {
                "model": MOCK_MODEL_ID,
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_abc123",
                                    "type": "function",
                                    "function": {
                                        "name": "foo_bar",
                                        "arguments": '{"param1":"baz"}',
                                    },
                                }
                            ],
                        },
                    },
                ],
            },
        ),
    ],
)
def test_chat_completion_responses_non_streaming(
    client,
    id_tracker,
    vllm_response,
    client_response,
):
    client_request = {
        "model": MOCK_MODEL_ID,
        "messages": [
            {
                "role": "system",
                "content": "system prompt",
            },
            {"role": "user", "content": "user prompt"},
        ],
    }
    response = client.chat.completions.create(**client_request)

    # ensure each response has a unique id
    assert response.id
    assert response.id not in id_tracker
    id_tracker.append(response.id)

    # Verify basic response attributes
    assert response.model == client_response["model"]
    assert response.created > 0
    assert response.object == "chat.completion"

    response_choices = response.choices
    for i, response_choice in enumerate(response_choices):
        # Assert indexes appear in the right order
        assert response_choice.index == i

        # Assert each response choice matches our expected json
        response_json = response_choice.model_dump(exclude_unset=True)
        expected_json = client_response["choices"][i]
        assert response_json == expected_json
