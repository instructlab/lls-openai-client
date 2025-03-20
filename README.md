# Llama Stack OpenAI Client

This library has two pieces - an OpenAI client that delegates all
calls to Llama Stack APIs and an OpenAI proxy server that delegates
all incoming requests to Llama Stack APIs.

## OpenAI client

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
