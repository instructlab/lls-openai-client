# Llama Stack OpenAI Client

This library has two pieces - an OpenAI client that delegates all
calls to Llama Stack APIs and an OpenAI proxy server that delegates
all incoming requests to Llama Stack APIs.

## OpenAI client

The OpenAI client adapter allows you to create an object that looks
and acts like a Python OpenAI client. But, under the hood, it
delegates all inference calls to the LlamaStack API.

### Usage

Install Llama Stack Server and the remote-vllm distribution's
dependent libraries:

```
pip install llama-stack aiosqlite autoevals datasets faiss-cpu mcp \
    opentelemetry-exporter-otlp-proto-http opentelemetry-sdk
```

Install this OpenAI client adapter:

```
pip install git+https://github.com/bbrowning/llama-stack-openai-client
```

Set the VLLM_URL and INFERENCE_MODEL environment variables as required
when using LlamaStack as a library client with the remote-vllm
provider. These should point towards your running vLLM server and the
model deployed in it.

```
export VLLM_URL=http://localhost:8000/v1
export INFERENCE_MODEL="meta-llama/Llama-3.2-3B-Instruct"
```

Run
```
import os
from llama_stack.distribution.library_client import LlamaStackAsLibraryClient
from lls_openai_client.client_adapter import OpenAIClientAdapter

# Create and initialize our Llama Stack client
lls_client = LlamaStackAsLibraryClient("remote-vllm")
lls_client.initialize()

# Wrap the Llama Stack client in an OpenAI client API
client = OpenAIClientAdapter(lls_client)

llama_prompt = """<|start_header_id|>user<|end_header_id|>

What is 2+2?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

# Use the typical OpenAI client calls to do inference
response = client.completions.create(
    model=os.environ["INFERENCE_MODEL"],
    prompt=llama_prompt,
)

print(f"\nResponse:\n{response.choices[0].text}")
```

## OpenAI proxy server

## Development

To setup your local development environment from a fresh clone of this
repository:

```
python -m venv venv
source venv/bin/activate
pip install -e . -r requirements-dev.txt
```

### Running unit tests

```
tox -e py3-unit
```

### Running functional tests

```
tox -e py3-functional
```

### Running lint, ruff, mypy, all tests

```
tox
```
