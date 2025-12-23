from pathlib import Path
from typing import Optional, Type, TypeVar

from jinja2 import BaseLoader, Environment, PackageLoader, select_autoescape

from ainterviewer.agents.prompts import agent_prompts
from ainterviewer.agents.prompts.agent_prompts import ProbingAgentPrompts
from ainterviewer.agents.prompts.models import BasePrompts
from ainterviewer.exceptions import LanguageNotSupportedError
from ainterviewer.types import LanguageCode

PROMPT_LANGS = [path.name for path in Path(__file__).parent.glob("templates/*/")]


def get_prompt_templates(lang: LanguageCode = "EN") -> dict[str, str]:
    package_loader = PackageLoader("ainterviewer.agents.prompts.templates", lang)

    env_package = Environment(loader=package_loader, autoescape=select_autoescape())

    # Get the names of the templates from the package
    template_names = package_loader.list_templates()

    # Read the template content and store it in a dictionary
    template_dict: dict[str, str] = {}
    for template_name in template_names:
        if not template_name.endswith((".jinja", ".jinja2")):
            continue
        template_dict[template_name] = package_loader.get_source(
            env_package, template_name
        )[0]

    return template_dict


T = TypeVar("T", bound=BasePrompts)


def get_agent_prompts(
    agent_name: str,
    lang: LanguageCode = "EN",
    template_loader: Optional[BaseLoader] = None,
    **kwargs,
) -> T:  # type: ignore
    if lang not in PROMPT_LANGS:
        raise LanguageNotSupportedError(f"Language {lang} not supported.")

    # Dynamically access the correct class from the module
    AgentPrompts: Type[T] = getattr(agent_prompts, f"{agent_name}Prompts")

    prompt_instance: T = AgentPrompts(
        template_loader=template_loader,
        lang=lang,
        **kwargs,
    )

    return prompt_instance


if __name__ == "__main__":
    probing_agent_prompts: ProbingAgentPrompts = get_agent_prompts(
        "ProbingAgent", lang="DA"
    )

    print(probing_agent_prompts.system_prompt)
