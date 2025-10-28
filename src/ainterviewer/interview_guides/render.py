import json
from typing import Any, Dict, List


class InterviewGuideTemplate:
    def __init__(self, interview_guide: Dict[str, Any]):
        self.interview_guide = interview_guide

    def render(self) -> str:
        return f"""## Framing
{self.interview_guide["framing"]}

## Introduction 
{self.interview_guide["introduction"]}

## Sections
{self._render_sections()}
{self._render_outro()}
""".strip()

    def _render_sections(self) -> str:
        return "\n\n".join(
            self._render_section(section)
            for section in self.interview_guide["question_sections"]
        )

    def _render_section(self, section: Dict[str, Any]) -> str:
        return f"""### Batttery
{section["description"]}

{self._render_questions(section["questions"])}"""

    def _render_questions(self, questions: List[Dict[str, Any]]) -> str:
        return "\n\n".join(self._render_question(question) for question in questions)

    def _render_question(self, question: Dict[str, Any]) -> str:
        # TODO: Add survey item and images

        probes = self._render_probes(question.get("probes", []))
        return f"""{question["main_question"]}{probes}"""

    def _render_probes(self, probes: List[str]) -> str:
        if not probes:
            return ""
        return "\n" + "\n".join(f"- {probe}" for probe in probes)

    def _render_outro(self):
        if outro := self.interview_guide.get("outro"):
            return f"""
## Outro
{outro}"""
        else:
            return ""


def load_interview_guide(file_path: str) -> Dict[str, Any]:
    with open(file_path, "r") as f:
        return json.load(f)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "interview_guide", help="The path to the interview guide JSON file"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    interview_guide = load_interview_guide(args.interview_guide)
    template = InterviewGuideTemplate(interview_guide)
    print(template.render())
