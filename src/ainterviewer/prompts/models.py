from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Optional, Union

from jinja2 import (
    BaseLoader,
    Environment,
    PackageLoader,
    StrictUndefined,
    Template,
)
from pydantic import BaseModel

from ainterviewer.types import LanguageCode

# TODO:
# The individual prompt templates should probably be validated in this models.
# That means checking for mandatory template placeholders and ensuring that the
# provided templates are valid jinja2 templates.


class Prompts(BaseModel):
    probing_agent: PromptTemplates
    classification_agent: PromptTemplates
    answering_agent: PromptTemplates
    history_agent: PromptTemplates
    security_agent: PromptTemplates
    translation_agent: PromptTemplates
    visual_agent: PromptTemplates
    extra_prompts: dict[str, str]

    def dump_templates(self) -> dict[str, str]:
        dump = {
            agent + "_" + template_name + ".jinja": template_content
            for agent, templates in self.__dict__.items()
            if "agent" in agent
            for template_name, template_content in templates.model_dump().items()
        }

        dump |= {
            prompt + "_prompt.jinja": content
            for prompt, content in self.extra_prompts.items()
        }

        return dump

    def print_prompts(self):
        for agent, templates in self.__dict__.items():
            if "agent" in agent:
                header = f"============== {agent} =============="
                print("=" * len(header))
                print(header)
                print("=" * len(header) + "\n")
                for template_name, template_content in templates.model_dump().items():
                    subheader = f"============== {template_name} =============="
                    print(subheader)
                    print(template_content)

                print("\n\n")


class PromptTemplates(BaseModel):
    system_prompt: str
    instruction_prompt: str


def get_default_prompts() -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    prompt_files = Path(__file__).parent.glob("templates/EN/*.jinja")

    agent_prompts: dict[str, dict[str, str]] = defaultdict(dict)

    extra_prompts: dict[str, str] = {}

    for prompt in prompt_files:
        if "agent" in prompt.stem:
            agent_name = "_".join(prompt.stem.split("_")[0:2])
            prompt_name = "_".join(prompt.stem.split("_")[2:4])
            agent_prompts[agent_name][prompt_name] = prompt.read_text()
        else:
            prompt_name = "_".join(prompt.stem.split("_")[:-1])
            extra_prompts[prompt_name] = prompt.read_text()

    return agent_prompts, extra_prompts


_agent_prompts, _extra_prompts = get_default_prompts()
DEFAULT_PROMPTS = Prompts(**_agent_prompts, extra_prompts=_extra_prompts)  # ty: ignore[invalid-argument-type]


class BasePrompts(ABC):
    system_prompt = ""

    def __init__(
        self,
        lang: LanguageCode = "EN",
        template_loader: Optional[BaseLoader] = None,
    ):
        if not template_loader:
            template_loader = PackageLoader("ainterviewer.prompts.templates", lang)

        self.env = Environment(loader=template_loader, undefined=StrictUndefined)

    def get_template(self, template_name: str) -> Template:
        return self.env.get_template(template_name)

    def get_source(self, template: Union[str, Template]) -> str:
        if isinstance(template, str):
            return self.env.loader.get_source(self.env, template)[0]
        elif isinstance(template, Template):
            if not template.filename:
                raise FileNotFoundError("The template has no filename")

            return self.env.loader.get_source(
                self.env, template.filename.split("/")[-1]
            )[0]

        raise TypeError(
            f"Expected `template` to be of type `str` or `Template`, but got {type(template)}"
        )

    @abstractmethod
    def generate_system_prompt(self) -> Union[str, Template]: ...

    def print_prompt(self):
        if self.system_prompt:
            print("============== SYSTEM PROMPT ==============")
            print(self.system_prompt)
            print("===========================================\n\n")
