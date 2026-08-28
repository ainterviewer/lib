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


def get_extra_model_kwargs(model: str, for_response_model: bool):
    extra_model_kwargs = {}

    if model == "openai/gpt-oss-120b":
        extra_model_kwargs["reasoning_effort"] = "low"
        extra_model_kwargs["extra_body"] = {"top_k": 3}

        if not for_response_model:
            # TODO:
            # - this should be model based and maybe also question based.
            # - maybe they should be words, and tokens fetched and cached from the api.
            # - how to implement in interface.
            print("Applying logit bias")
            extra_model_kwargs["logit_bias"] = {
                # Negative
                "4157": -3,  #  kun
                "65512": -7,  # Kan
                "98936": -5,  # Kunne
                "11": -7,  # ,
                "80750": -15,  # konkre
                "102719": -15,  #  konkret
                "12855": -15,  #  specif
                # Positive
                "73760": 5,  # Tak
                "30": 7,  # ?
            }

    return extra_model_kwargs
