"""Either probing limit at 0 turns probing off, whatever the other says."""

import pytest

from ainterviewer.interview import AInterviewer
from ainterviewer.interview_guides import Question
from ainterviewer.interview_guides.localized import LocalizedQuestion


@pytest.mark.parametrize(
    ("max_probes_n", "max_probes_time", "probes"),
    [
        (3, None, True),
        (None, 60, True),
        (3, 60, True),
        (0, None, False),
        (None, 0, False),
        (3, 0, False),
        (0, 60, False),
        (0, 0, False),
        (None, None, False),
    ],
)
def test_either_limit_at_zero_turns_probing_off(max_probes_n, max_probes_time, probes):
    question = Question(
        main_question="How are you?",
        max_probes_n=max_probes_n,
        max_probes_time=max_probes_time,
    )
    assert AInterviewer.should_probe(question) is probes


def test_zero_limits_validate():
    for n, time in [(0, 0), (0, 60), (3, 0)]:
        LocalizedQuestion.model_validate(
            {"main_question": {"EN": "x"}, "max_probes_n": n, "max_probes_time": time}
        ).render({}, ["EN"])
