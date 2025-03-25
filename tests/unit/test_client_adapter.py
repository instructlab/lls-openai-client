# SPDX-License-Identifier: Apache-2.0

# First Party
# pylint: disable=import-error
from lls_openai_client.client_adapter import _parse_request_response_format


def test_guided_choice_response_format():
    choices = ["foo", "bar", "baz"]
    params = {"extra_body": {"guided_choice": choices}}
    response_fmt = _parse_request_response_format(params)
    assert response_fmt["type"] == "json_schema"
    for choice in choices:
        assert choice in response_fmt["json_schema"]["pattern"]
