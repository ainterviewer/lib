"""Guides the interview loop runs without a language model."""

import uuid

from ainterviewer.interview_guides import InterviewGuide


def make_guide() -> InterviewGuide:
    """An introduction, a free-text question, a survey question and an outro,
    none of which needs a language model."""
    return InterviewGuide.model_validate(
        {
            "introduction": "Hello",
            "outro": "Goodbye",
            "question_sections": [
                {
                    "description": "Section",
                    "questions": [
                        {
                            "main_question": "Free?",
                            "max_probes_n": 0,
                            "check_if_answered": False,
                            "create_segue": False,
                        },
                        {
                            "main_question": "Agree?",
                            "survey_item": {
                                "type": "likert",
                                "options": ["No", "Yes"],
                            },
                        },
                    ],
                }
            ],
        }
    )


def two_sections(action: str) -> InterviewGuide:
    """A first question whose answer "stop" triggers `action` before the
    second is asked, and a section after it."""
    first = uuid.UUID(int=1)
    plain = {"max_probes_n": 0, "check_if_answered": False, "create_segue": False}
    return InterviewGuide.model_validate(
        {
            "question_sections": [
                {
                    "description": "Work",
                    "questions": [
                        {"id": str(first), "main_question": "Go on?", **plain},
                        {
                            "main_question": "Why?",
                            "conditions": {
                                "action": action,
                                "conditions": [
                                    {
                                        "question_context": {
                                            "question_id": str(first),
                                            "question_index": [0, 0],
                                        },
                                        "evaluation": [{"trigger_value": "stop"}],
                                    }
                                ],
                            },
                            **plain,
                        },
                    ],
                },
                {
                    "description": "Home",
                    "questions": [{"main_question": "At home?", **plain}],
                },
            ]
        }
    )
