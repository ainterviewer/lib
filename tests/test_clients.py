from ainterviewer.lpm.clients import openrouter_model_kwargs


def test_gpt_oss_prefers_deepinfra():
    assert openrouter_model_kwargs("openrouter:openai/gpt-oss-120b") == {
        "extra_body": {"provider": {"order": ["deepinfra"]}}
    }


def test_other_models_use_openrouter_routing():
    assert openrouter_model_kwargs("openrouter:z-ai/glm-5.3-flash") == {}
