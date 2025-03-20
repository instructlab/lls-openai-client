# SPDX-License-Identifier: Apache-2.0

# pylint: disable=redefined-outer-name, unused-argument

# Standard
from unittest.mock import AsyncMock, patch

# Third Party
import pytest


@pytest.fixture
def mock_openai_chat_completion():
    with patch(
        "openai.resources.chat.completions.AsyncCompletions.create",
        new_callable=AsyncMock,
    ) as mock_completion:
        yield mock_completion


def test_chat_completion_messages_text_type(
    client, model_id, mock_openai_chat_completion
):
    messages = [{"role": "user", "content": [{"type": "text", "text": "foo"}]}]
    args = []
    kwargs = {
        "model": model_id,
        "messages": messages,
    }
    response = client.chat.completions.create(*args, **kwargs)
    mock_openai_chat_completion.assert_called()
    call_kwargs = mock_openai_chat_completion.call_args.kwargs
    assert call_kwargs["model"] == model_id
    assert call_kwargs["messages"] == messages
    assert response.choices
    assert response.choices[0].message
    # TODO: actually test response itself as well...
