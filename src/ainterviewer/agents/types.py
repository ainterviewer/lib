from enum import StrEnum


class DiceStrategy(StrEnum):
    DESCRIPTIVE = "descriptive"
    IDIOGRAPHIC = "idiographic"
    CLARIFYING = "clarifying"
    EXPLANATORY = "explanatory"


class ProbingStrategy(StrEnum):
    STANDARD = "standard"
    DICE_MASTER_TO_ONE_PROBE = "dice_master_to_one_probe"
    DICE_ENSEMBLE_TO_MASTER_PROBE = "dice_ensemble_to_master_probe"
    DICE_MASTER_TO_ENSEMBLE_TO_ONE_PROBE = "dice_master_to_ensemble_to_one_probe"
