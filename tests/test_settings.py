import pytest
from pydantic import ValidationError

from ainterviewer.settings import LLMSettings


def test_vllm_api_base_defaults_to_host_and_port():
    llm = LLMSettings(llm_host="10.0.0.5", llm_port=8000)
    assert llm.vllm_api_base == "http://10.0.0.5:8000/v1"


def test_vllm_base_url_overrides_host_and_port():
    llm = LLMSettings(
        llm_host="10.0.0.5", vllm_base_url="https://llm.example.org/vllm/v1/"
    )
    assert llm.vllm_api_base == "https://llm.example.org/vllm/v1"


def test_default_model_goes_through_openrouter():
    # Without a provider prefix a model is sent to the vLLM server instead.
    assert LLMSettings().default_model.startswith("openrouter:")


def test_available_models_lists_distributable_models_first():
    llm = LLMSettings(models=["a"], test_only_models=["b"], default_model="b")
    assert llm.available_models == ["a", "b"]


def test_a_model_in_neither_list_is_test_only():
    llm = LLMSettings(models=["a"], test_only_models=["b"], default_model="a")
    assert not llm.is_test_only("a")
    assert llm.is_test_only("b")
    assert llm.is_test_only("unlisted")


def test_default_model_may_be_test_only():
    llm = LLMSettings(models=["a"], test_only_models=["b"], default_model="b")
    assert llm.is_test_only(llm.default_model)


def test_model_lists_cannot_overlap():
    with pytest.raises(ValidationError, match="overlap"):
        LLMSettings(models=["a"], test_only_models=["a"], default_model="a")


def test_available_models_is_refused_rather_than_ignored():
    with pytest.raises(ValidationError, match="test_only_models"):
        LLMSettings.model_validate({"available_models": ["a"], "default_model": "a"})
