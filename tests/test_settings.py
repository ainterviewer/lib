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
