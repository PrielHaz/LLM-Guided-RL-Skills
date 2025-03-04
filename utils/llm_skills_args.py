import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from typing import List, Optional

from nlp_module.option_policy import OptionPolicy


# has description and name of the skill
class Skill:
    """
    Class to represent a skill.
    """

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    # cut desc to desc_max_length chars then .....
    def __str__(self, desc_max_length=100):
        return f"Skill(name={self.name}, description={self.description[:desc_max_length] + '...'})"


class LLM_Skills_Args:
    """
    Class to hold arguments for LLM-based skill augmentation in CrafterGymnasium.
    """

    def __init__(
        self,
        skills: List[Skill],
        option_policy: OptionPolicy,
        num_steps_pass_llm: int = 3,
        default_action_index: int = 5,
    ):
        """
        Initializes the LLM_Skills_Args object.

        Args:
            skills (List[Skill]): List of skills to be used in the option policy.
            option_policy (OptionPolicy): LLM-based option policy object.
            num_steps_pass_llm (int): Number of previous steps to include in LLM prompt.
            default_action_index (int): Index of the default action to be used in the option policy.
        """
        self.skills = skills
        self.option_policy = option_policy
        self.num_steps_pass_llm = num_steps_pass_llm
        self.default_action_index = default_action_index
