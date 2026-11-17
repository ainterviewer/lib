import pytest
from pydantic import ValidationError

from ainterviewer.settings import (
    ApiKeys,
    ConfigError,
    ConfigSettings,
    ConfigWarning,
    LLMSettings,
    Settings,
)


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


# config.toml holds no secrets, so it can be committed.


@pytest.fixture
def config_toml(monkeypatch, tmp_path):
    """Write a config.toml into an otherwise empty working directory."""
    monkeypatch.chdir(tmp_path)

    def write(text: str) -> None:
        (tmp_path / "config.toml").write_text(text)

    return write


class _App(ConfigSettings):
    """Settings that know every other reader of their file."""

    shares_file_with = (Settings,)

    debug: bool = False
    app: dict | None = None


@pytest.mark.parametrize(
    "toml",
    [
        'openai_api_key = "sk-..."',
        '[llm]\nvllm_api_key = "x"',
        '[services.email.sender]\npassword = "x"',
        'jwt_secret_key = "x"',
    ],
)
def test_a_secret_in_config_toml_is_refused(config_toml, toml):
    config_toml(toml)

    with pytest.raises(ConfigError, match="environment and .env only"):
        Settings()


def test_a_known_secret_names_its_variable(config_toml):
    config_toml('openrouter_api_key = "x"')

    with pytest.raises(ConfigError, match="OPENROUTER_API_KEY"):
        Settings()


def test_a_secrets_table_is_refused(config_toml):
    config_toml('[secrets]\nfoo = "x"')

    with pytest.raises(ConfigError, match=r"no \[secrets\] table"):
        Settings()


def test_api_keys_are_not_read_from_config_toml(config_toml, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config_toml("debug = true")
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")

    assert ApiKeys().openai_api_key.get_secret_value() == "from-env"


def test_an_unknown_key_in_a_table_is_refused(config_toml):
    config_toml('[llm]\ndefualt_model = "x"')

    with pytest.raises(ValidationError, match="defualt_model"):
        Settings()


def test_an_unknown_top_level_key_is_warned_about(config_toml):
    config_toml("[app]\n[llm]\n[storage]\n[servics]")

    with pytest.warns(ConfigWarning, match="nothing reads servics;"):
        _App()


def test_tables_of_the_other_readers_are_not_warned_about(config_toml, recwarn):
    config_toml("debug = true\n[app]\n[llm]")

    _App()
    Settings()

    assert not [w for w in recwarn if issubclass(w.category, ConfigWarning)]
