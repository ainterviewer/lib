from __future__ import annotations

import warnings
from typing import Literal, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    field_validator,
    model_validator,
)

from ainterviewer.settings import settings

# https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/__init__.py#L9
QuantizationMethods = Literal[
    "awq",
    "deepspeedfp",
    "tpu_int8",
    "fp8",
    "ptpc_fp8",
    "fbgemm_fp8",
    "modelopt",
    "modelopt_fp4",
    "bitblas",
    "gguf",
    "gptq_marlin_24",
    "gptq_marlin",
    "gptq_bitblas",
    "awq_marlin",
    "gptq",
    "compressed-tensors",
    "bitsandbytes",
    "hqq",
    "experts_int8",
    "neuron_quant",
    "ipex",
    "quark",
    "moe_wna16",
    "torchao",
    "auto-round",
    "rtn",
    "inc",
    "mxfp4",
    "petit_nvfp4",
]


class VLLMModelConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    model: str

    # NOTE:
    # Engine args from
    # https://docs.vllm.ai/en/latest/serving/engine_args.html
    # see vllm.engine.arg_utils

    tokenizer: str | None = None
    tokenizer_mode: str | None = None
    dtype: str | None = None
    gpu_memory_utilization: float | None = Field(0.95, ge=0, le=1)
    max_model_len: Optional[int] = None
    max_num_seq: Optional[int] = 20
    max_seq_len_to_capture: Optional[int] = None
    quantization: Optional[QuantizationMethods] = None
    load_format: Optional[str] = None
    enforce_eager: Optional[bool] = None
    enable_chunked_prefill: Optional[bool] = True
    served_model_name: Optional[str] = None
    config_format: Literal["auto", "hf", "mistral"] | None = None
    limit_mm_per_prompt: dict | None = None
    mm_preprocessor_cache_gb: int | None = None

    @field_validator("model")
    def validate_model(cls, model: str) -> str:
        if settings.llm.model_storage == "s3_bucket":
            return "s3://ainterviewer-sodas/data/llms/" + model.split("/")[-1]
        return model

    @field_validator("load_format")
    def validate_load_format(cls, load_format: str) -> str | None:
        if settings.llm.model_storage == "s3_bucket":
            if load_format is not None:
                warnings.warn(
                    f"load_format == {load_format} is overwritten to 'runai_streamer' due to {settings.llm.model_storage=}"
                )
            return "runai_streamer"
        return load_format

    @model_validator(mode="after")
    def validate_config(self):
        if self.max_seq_len_to_capture is not None and self.max_model_len is not None:
            self.max_seq_len_to_capture = min(
                self.max_seq_len_to_capture, self.max_model_len
            )
        else:
            self.max_seq_len_to_capture = self.max_model_len
        return self


class VLLMModelConfigs(RootModel):
    root: dict[str, VLLMModelConfig]

    def __getitem__(self, model: str) -> VLLMModelConfig:
        return self.root[model]

    def get(self, model: str) -> Optional[VLLMModelConfig]:
        try:
            return self.root.get(model)
        except KeyError:
            return


VLLM_MODEL_CONFIGS = VLLMModelConfigs(
    **{
        "gemma3-27b": {
            "model": "pytorch/gemma-3-27b-it-FP8",  # leon-se/gemma-3-27b-it-FP8-Dynamic ISTA-DASLab/gemma-3-27b-it-GPTQ-4b-128g
            "served_model_name": "google/gemma-3-27b-it",
            "max_model_len": 12000,
            "enforce_eager": True,
            # "max_num_seq": 32,
            "limit_mm_per_prompt": {"image": 0, "video": 0},
            "mm_preprocessor_cache_gb": 0,
        },
        "mistral-small": {
            "model": "stelterlab/Mistral-Small-3.2-24B-Instruct-2506-FP8",
            "served_model_name": "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
            "max_model_len": 8000,
            # "max_num_seq": 512,
            "config_format": "mistral",
            "tokenizer_mode": "mistral",
            "load_format": "mistral",
            "limit_mm_per_prompt": {"image": 0, "video": 0},
            "mm_preprocessor_cache_gb": 0,
        },
        "gpt-oss-120b": {
            "model": "openai/gpt-oss-120b",
            "served_model_name": "gpt-oss-120b",
            "max_model_len": 8000,
            # "max_num_seq": 512,
            "limit_mm_per_prompt": {"image": 0, "video": 0},
            "mm_preprocessor_cache_gb": 0,
        },
    }
)

if __name__ == "__main__":
    print(VLLM_MODEL_CONFIGS)
