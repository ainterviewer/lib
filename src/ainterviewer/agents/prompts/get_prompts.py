from pathlib import Path
from typing import Type, TypeVar

from jinja2 import BaseLoader, Environment, PackageLoader, select_autoescape

from ainterviewer.agents.prompts import agent_prompts
from ainterviewer.agents.prompts.agent_prompts import BasePrompts
from ainterviewer.constants import LANGUAGE_CODES
from ainterviewer.exceptions import LanguageNotSupportedError
from ainterviewer.types import LanguageCode

# The languages that have their own prompt template directory. This is *not* the set
# of supported interview languages: the agents always render the English templates and
# are told which language to speak through the `translation` prompt variable
# (see `BasePrompts.translation`), so any code in `LANGUAGE_CODES` works.
#
# Nothing reads this today. It is kept for a future reintroduction of per-language
# template directories, which would need to fall back to `EN` for any language without
# its own directory.
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
    template_loader: BaseLoader | None = None,
    **kwargs,
) -> T:
    lang = lang.upper()

    if lang not in LANGUAGE_CODES:
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
    my_agent_prompts = get_agent_prompts("ReformulationAgent", lang="EN")

    print(my_agent_prompts.system_prompt)
