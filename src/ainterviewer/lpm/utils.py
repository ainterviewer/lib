import warnings
from functools import cache

import requests

from ainterviewer.types import ClassificationResponse, MessageRole


def map_system_to_user(message_role: MessageRole) -> MessageRole:
    """Maps system role to user role for models that do not support system role."""
    return message_role if message_role != MessageRole.SYSTEM else MessageRole.USER


@cache
def tokenize(model, text, base_api="http://localhost:8667/tokenize"):
    response = requests.post(
        base_api,
        json={"model": model, "prompt": text},
    )
    response.raise_for_status()
    return response.json()


def get_classification_response_tokens(model):
    if model.startswith("gpt"):
        return
    response_tokens = {}

    for response in ClassificationResponse:
        result = tokenize(model, response.value)
        tokens = result["tokens"][-1:]  # Skip first since it's begging of text token
        if len(tokens) > 1:
            warnings.warn("Expected a single token in the response, but got multiple")
        response_tokens[response.value] = tokens

    return response_tokens
