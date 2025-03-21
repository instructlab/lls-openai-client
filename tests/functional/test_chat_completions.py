# SPDX-License-Identifier: Apache-2.0

# pylint: disable=redefined-outer-name, unused-argument

# Standard
from unittest.mock import MagicMock, patch
import json

# Third Party
import httpx
import pytest


@pytest.fixture
def mock_httpx_send(model_id):
    with patch("httpx.AsyncClient.send") as mock_send:
        request = MagicMock()
        request.headers = {}
        mock_send.return_value = httpx.Response(
            status_code=200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "mock response",
                        },
                    }
                ],
                "model": model_id,
            },
            request=request,
        )
        yield mock_send


# list scoped to the entire module so we can track ids seen across
# responses
@pytest.fixture(scope="module")
def id_tracker():
    return []


@pytest.mark.parametrize(
    "create_params,expected_request_json",
    [
        (
            {
                "model": "foo",
                "messages": [{"role": "user", "content": "user prompt"}],
                "n": 5,
            },
            {
                "model": "foo",
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
                "model": "foo",
                "messages": [
                    {
                        "role": "system",
                        "content": "system prompt",
                    },
                    {"role": "user", "content": "user prompt"},
                ],
            },
            {
                "model": "foo",
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
                "model": "foo",
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "user prompt"}],
                    }
                ],
            },
            {
                "model": "foo",
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
                "model": "foo",
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "user prompt"}],
                    }
                ],
                "max_tokens": 8192,
                "temperature": 0.5,
            },
            {
                "model": "foo",
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
                "max_tokens": 8192,
                "stream": False,
                "temperature": 0.5,
                "tools": None,
                "top_p": 1.0,
            },
        ),
        (
            {
                "model": "foo",
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "user prompt"}],
                    }
                ],
                "top_p": 0.8,
            },
            {
                "model": "foo",
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
                "model": "foo",
                "messages": [{"role": "user", "content": "user prompt"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "user function",
                        },
                    }
                ],
            },
            {
                "model": "foo",
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
                            "name": "user function",
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
        (
            {
                "model": "foo",
                "messages": [{"role": "user", "content": "user prompt"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "user function",
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
                "model": "foo",
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
                            "name": "user function",
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
    ],
)
def test_chat_completion_requests_non_streaming(
    client,
    model_id,
    mock_httpx_send,
    id_tracker,
    create_params,
    expected_request_json,
):
    response = client.chat.completions.create(**create_params)

    mock_httpx_send.assert_called()
    send_calls = mock_httpx_send.call_args_list
    num_expected_calls = create_params.get("n", 1)
    assert len(send_calls) == num_expected_calls

    for send_call in send_calls:
        request = send_call.args[0]
        assert isinstance(request, httpx.Request)
        request_json = json.loads(request.content)
        assert request_json == expected_request_json

    # ensure each response has a unique id
    assert response.id not in id_tracker
    id_tracker.append(response.id)
