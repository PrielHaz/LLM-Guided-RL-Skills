import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List

import cv2
import numpy as np

from utils import util_funcs
from utils.llm_skills_args import Skill


def load_skills(skills_descriptions_dir):
    """Loads the skills from the given directory.

    Args:
        skills_descriptions_dir: The directory containing the skill descriptions.

    Returns:
        List[Skill]: A list of Skills.
    """
    skills: List[Skill] = []
    # iterate skills dir and create LLM_Skills_Args
    for skill_file in os.listdir(skills_descriptions_dir):
        skill_file_path = os.path.join(skills_descriptions_dir, skill_file)
        skill_desc = open(skill_file_path, "r").read()
        skill_name = util_funcs.get_skill_name_from_desc(skill_desc)
        skills.append(Skill(name=skill_name, description=skill_desc))
    return skills


class logger:
    def __init__(self, verbose, print_file_path=None):
        """
        Initializes the logger object.

        Args:
            verbose (int): Verbosity level of the logger.
            print_file_path (str): Path to a file to print log messages to. If None, messages are printed to the console.
        """
        self.verbose = verbose
        self.print_file_path = print_file_path
        # if exists, clear the file:
        if self.print_file_path:
            with open(self.print_file_path, "w") as f:
                f.write("")
        self.log(f"Logger initialized with verbosity level {verbose}.")
        if self.print_file_path:
            self.log(f"Printing log messages to file: {self.print_file_path}.")
        else:
            self.log("Printing log messages to console.")

    def log(self, message, level=1):
        """
        Logs a message to the console or a file if verbosity level is high enough.

        Args:
            message (str): The message to log.
            level (int): Required verbosity level to print this message.
        """
        if self.verbose < level:
            return

        if not self.print_file_path:
            print(message)
            return
        # now there is a file to print to:
        try:
            with open(self.print_file_path, "a") as f:
                f.write(message + "\n")
        except:
            error_message = f"[Logger Error] Exception occurred while appending to the log file, maybe JSONDecodeError"
            with open(self.print_file_path, "a") as f:
                f.write(error_message + "\n")


# sep by ' then take the [1] elem:
def get_skill_name_from_desc(skill_desc: str):
    """Extracts the skill name from its description."""
    return skill_desc.split("'")[1]


def annotate_obs(
    obs,
    action_text,
    reward_text,
    padding_w=160,  # 160 + 64=224 divisible by 16 so no imageio warnings
    font_size=0.4,
    font_thickness=1,
    text_color=(255, 255, 255),
    bg_color=(0, 0, 0),
):
    """Annotates an observation image with action and reward text."""
    obs_h, obs_w, _ = obs.shape
    # Convert RGB to BGR (for OpenCV)
    obs_bgr = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)

    # Create a padded area on the right
    padded_obs = np.zeros((obs_h, obs_w + padding_w, 3), dtype=np.uint8)
    padded_obs[:, :obs_w, :] = obs_bgr  # Copy original obs

    # Add black background on the right side
    cv2.rectangle(padded_obs, (obs_w, 0), (obs_w + padding_w, obs_h), bg_color, -1)

    # Text properties
    text_x = obs_w + 5  # Left margin for text
    cv2.putText(
        padded_obs,
        action_text,
        (text_x, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_size,
        text_color,
        font_thickness,
        cv2.LINE_AA,
    )
    cv2.putText(
        padded_obs,
        reward_text,
        (text_x, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_size,
        text_color,
        font_thickness,
        cv2.LINE_AA,
    )

    return padded_obs


def get_prompt_messages_nicely(messages):
    """

    Args:
        messages (list): A list of dictionaries, where each dictionary contains 'role' and 'content' keys.
    """
    if not messages:
        return "No messages to display."
    output = ""
    for i, message in enumerate(messages):
        # Print role in uppercase with formatting
        role = message.get("role", "unknown").upper()
        output += f"MESSAGE #{i+1}, ROLE: {role}, CONTENT:\n"
        # Print content with proper indentation
        content = message.get("content", "No content")
        for line in content.split("\n"):
            output += f"    {line}\n"

        # Add extra space between messages
        output += "\n\n"

    return output
