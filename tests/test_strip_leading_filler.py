import pytest

from ainterviewer.utils import strip_leading_filler


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "Er er der andre forhold eller funktioner, som også spiller en rolle?",
            "Er der andre forhold eller funktioner, som også spiller en rolle?",
        ),
        (
            "Er er der andre faktorer eller hensyn, som du også lægger vægt på, når du vælger en feriedestination?",
            "Er der andre faktorer eller hensyn, som du også lægger vægt på, når du vælger en feriedestination?",
        ),
        (
            "Er kan du nævne nogle konkrete ting, du typisk tænker på eller lægger vægt på, når du skal vælge en ny telefon?",
            "Kan du nævne nogle konkrete ting, du typisk tænker på eller lægger vægt på, når du skal vælge en ny telefon?",
        ),
        ("Er hvad var grunden til det?", "Hvad var grunden til det?"),
        ("Er, hvordan oplevede du det?", "Hvordan oplevede du det?"),
        ("Er… kan du uddybe det?", "Kan du uddybe det?"),
        ("Er Hvorfor valgte du den?", "Hvorfor valgte du den?"),
        ("  Er er det vigtigt for dig?", "  Er det vigtigt for dig?"),
    ],
)
def test_strips_danish_filler(text, expected):
    assert strip_leading_filler(text, "da") == expected


@pytest.mark.parametrize(
    "text",
    [
        "Er der andre forhold, der spiller en rolle?",
        "Er det vigtigt for dig?",
        "Er du tilfreds med telefonen?",
        "Erfaringer med mobiltelefoner – hvad har du?",
        "Hvad var grunden til det?",
        "er hvad",
        "Er",
    ],
)
def test_keeps_genuine_danish(text):
    assert strip_leading_filler(text, "DA") == text


def test_only_applies_to_danish():
    assert strip_leading_filler("Er, what made you choose it?", "EN") == (
        "Er, what made you choose it?"
    )
