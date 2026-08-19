from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel, ConfigDict

# TODO:
# The individual prompt templates should probably be validated in this models.
# That means checking for mandatory template placeholders and ensuring that the
# provided templates are valid jinja2 templates.


class Prompts(BaseModel):
    probing_agent: PromptTemplates
    classification_agent: PromptTemplates
    guide_agent: PromptTemplates
    answering_agent: PromptTemplates
    history_agent: PromptTemplates
    security_agent: PromptTemplates
    visual_agent: PromptTemplates
    reformulation_agent: PromptTemplates
    extra_prompts: dict[str, str]

    def dump_templates(self) -> dict[str, str]:
        dump = {
            agent + "/" + template_name + ".jinja": template_content
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
    model_config = ConfigDict(extra="allow")

    system_prompt: str
    instruction_prompt: str


def get_default_prompts() -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    templates_dir = Path(__file__).parent / "templates" / "EN"

    agent_prompts: dict[str, dict[str, str]] = defaultdict(dict)
    extra_prompts: dict[str, str] = {}

    # Agent prompts live in subdirectories: templates/EN/{agent_name}/{prompt}.jinja
    for prompt in templates_dir.glob("*_agent/*.jinja"):
        agent_name = prompt.parent.name
        prompt_name = prompt.stem
        agent_prompts[agent_name][prompt_name] = prompt.read_text()

    # Extra prompts live at the top level: templates/EN/{prompt}.jinja
    for prompt in templates_dir.glob("*.jinja"):
        if "agent" not in prompt.stem:
            prompt_name = "_".join(prompt.stem.split("_")[:-1])
            extra_prompts[prompt_name] = prompt.read_text()

    return agent_prompts, extra_prompts


_agent_prompts, _extra_prompts = get_default_prompts()
DEFAULT_PROMPTS = Prompts(**_agent_prompts, extra_prompts=_extra_prompts)
