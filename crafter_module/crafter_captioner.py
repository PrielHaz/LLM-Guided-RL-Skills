# To run the pip installed version:
# python3 -m crafter.run_gui --view 9 9 --area 64 64 --window 600 600 --size 0 0

# --view 32 32 make player see 32x32 grid elements around him
# --area 64 64 make the grid 64x64 elems so if run:
# python3 -m crafter.run_gui --view 8 8 --area 8 8 --size 200 200
# you see from the start all the grid, and 1 movement cause see the padding

# To run the local version run in the crafter dir:
# python run_gui.py

import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
import crafter
import numpy as np
import stable_baselines3

from nlp_module import prompts_builder
from utils import constants


# * Attention: all env in this file refers to raw crafter env,
# which can be accessed by CrafterGymnasium.env property.
class StateCaptioner:
    """Generates a textual description of the current game state."""

    def __init__(self, env: crafter.Env):
        assert isinstance(env, crafter.Env), "env must be an instance of crafter.Env."
        self.env = env

    def caption(
        self,
        state_info,
        size_around_player_to_caption=None,
        mask_objects_list=["grass"],
        explain_grid_format=True,
        explain_achieved=True,
        explain_not_achieved=True,
    ):
        """
        Generates a textual description of the current game state.
        size_around_player_to_caption: if None, defualts to the player view
        """
        inventory = state_info.get("inventory")
        player_pos = state_info.get("player_pos")
        achievements = state_info.get("achievements")
        semantic_map = state_info.get("semantic")

        player = self.env._player
        assert player is not None, "Player object not found in the environment."
        # def texture(self):
        # if self.sleeping:
        #     return "player-sleep"
        # return {
        #     (-1, 0): "player-left",
        #     (+1, 0): "player-right",
        #     (0, -1): "player-up",
        #     (0, +1): "player-down",  # key is (0,1) and value is "player_down"
        # }[tuple(self.facing)]
        texture = player.texture  # tells if asleep or facing direction

        # calc what is on front of the player:
        front_pos = player_pos + player.facing
        # if its outside area, write it's padding
        if not (
            0 <= front_pos[0] < self.env._area[0]
            and 0 <= front_pos[1] < self.env._area[1]
        ):
            front_material, front_obj = "padding", "padding"
        else:
            front_material, front_obj = self.env._world[front_pos]

        front_desc = front_obj.__class__.__name__ if front_obj else front_material

        # Inventory caption
        # inventory_caption = (
        #     ", ".join([f"{v} {k}" for k, v in inventory.items() if v > 0]) or "empty"
        # )
        inventory_caption = (
            ", ".join([f"{v} {k}" for k, v in inventory.items()]) or "empty"
        )
        # print each achievement and if have it or not:
        achieved_caption = ""
        not_achieved_caption = ""
        for k, v in achievements.items():
            if v > 0:
                achieved_caption += f"{k}, "
            else:
                not_achieved_caption += f"{k}, "
        # delete the last ", "
        if achieved_caption:
            achieved_caption = achieved_caption[:-2] + "."
        if not_achieved_caption:
            not_achieved_caption = not_achieved_caption[:-2] + "."
        if not achieved_caption:
            achieved_caption = (
                "No achievements yet :( complete achievements to gain reward!"
            )

        achievements_caption = ""
        if explain_achieved:
            achievements_caption += f"Achieved: {achieved_caption}"

        if explain_not_achieved:
            achievements_caption += f"\n\n\nNot achieved yet: {not_achieved_caption}"
        # Grid Caption
        grid_caption = GridCaptioner(self.env).caption(
            semantic_map,
            player_pos,
            size_around_player_to_caption,
            mask_objects_list=mask_objects_list,
            explain_grid_format=explain_grid_format,
        )
        return (
            f"State:\n"
            + f"texture: {texture}\n"
            + f"Inventory: {inventory_caption}\n"
            + f"{achievements_caption}\n"  # may be empty
            + f"Grid:\n {grid_caption}\n\n"
            + f"In front of player: {front_desc}"
        )


class GridCaptioner:
    """Describes the grid layout around the player, either in absolute or relative terms."""

    def __init__(self, env):
        assert isinstance(env, crafter.Env), "env must be an instance of crafter.Env."
        self.env = env

    def caption(
        self,
        semantic_map,
        player_pos,
        size_around_player_to_caption=None,
        mask_objects_list=None,
        explain_grid_format=True,
    ):
        """Generates a textual description of the grid layout around the player."""
        if semantic_map is None:
            return "No grid information available."

        # Set default masking if none provided
        mask_objects_list = mask_objects_list or []

        # Determine the size to caption
        if size_around_player_to_caption is None:
            size_x, size_y = self.env._view  # Default: observed view (e.g., 9x9)
        elif size_around_player_to_caption == "all":
            size_x, size_y = self.env._area  # Full semantic map (e.g., 64x64)
        else:
            size_x, size_y = size_around_player_to_caption  # Custom size

        half_size_x, half_size_y = size_x // 2, size_y // 2
        player_x, player_y = player_pos

        output = []
        if explain_grid_format:
            output.append(
                "The format is (x_relative, y_relative) where x_relative=2 means 2 steps right from the player, "
                "and y_relative=-2 means 2 steps below the player.\n"
            )
            output.append(
                "We start from top left corner of the grid, and move to the right, and then move to the next line.\n"
            )
            # Tell that all cells that we dont say what they are, are: mask_objects_list.join(", ")
            output.append(
                f"Note: All cells not mentioned are {', '.join(mask_objects_list)} marked as G instead of the regular format.\n"
            )

        for y in range(player_y - half_size_y, player_y + half_size_y + 1):
            line_entries = []
            for x in range(player_x - half_size_x, player_x + half_size_x + 1):
                if 0 <= x < self.env._area[0] and 0 <= y < self.env._area[1]:
                    material, obj = self.env._world[(x, y)]

                    # Compute relative coordinates
                    dx, dy = x - player_x, player_y - y

                    # Mark the player's position explicitly
                    if dx == 0 and dy == 0:
                        entry = f"({dx},{dy})=Player"
                    elif material in mask_objects_list:
                        # Can write hidden
                        # entry = f"({dx},{dy})=hidden"
                        entry = f"G"
                        # continue  # maybe we do want everything to put in the prompt... its not long
                    else:
                        obj_desc = obj.__class__.__name__ if obj else material
                        entry = f"({dx},{dy})={obj_desc}"

                    line_entries.append(entry)

            output.append(", ".join(line_entries))
            # If want to say we move to next line:
            # if y != player_y + half_size_y:
            #     # output.append("Moving to next line...\n")  # Indicate line break
            #     output.append("next grid line\n")  # Indicate line break

        return "\n".join(output)


# crafter_actions_map = {
#     "noop": {"index": 0, "requirement": "Always applicable."},
#     "move_left": {"index": 1, "requirement": "Flat ground lef
class ActionCaptioner:
    """Maps action indices to action descriptions."""

    def __init__(self, action_map):
        """
        Initializes the ActionCaptioner with a given mapping of actions.

        :param action_map: Dictionary mapping action names to their respective details,
                               including index and requirement.
        """
        self.action_map = action_map

    def caption(self, action_index, action_token_desc=True):
        """
        Returns a formatted caption for a given action index.

        :param action_index: Integer index of the action.
        :param action_token_desc: If True, use alphabet letters for the action index.
        :return: Formatted action description string.
        """
        action_tokens = self.get_actions_tokens_list()
        for action_name, action_info in self.action_map.items():
            if action_info["index"] == action_index:
                prefix = f"Action index={action_index}: "
                if action_token_desc:
                    letter = action_tokens[action_index]
                    prefix = f"Action {letter}: "
                return (
                    f"{prefix}{action_name}, Requirement: {action_info['requirement']}"
                )
        raise ValueError(f"Invalid action index: {action_index}")

    def num_actions(self):
        """
        Returns the number of actions in the mapping.

        :return: Integer number of actions.
        """
        return len(self.action_map)

    def caption_all(self, mask_indices=None):
        """
        Returns formatted descriptions for all actions.

        :return: String containing formatted descriptions of all actions.
        """
        if mask_indices is None:
            mask_indices = []
        output = []
        for idx in range(self.num_actions()):
            if idx in mask_indices:
                continue
            output.append(self.caption(idx))
        return "\n".join(output)

    def get_actions_tokens_list(self, mask_indices=None):
        """
        Returns a list of action tokens (like 'A', 'B', etc.), excluding masked indices.
        """
        tokens = [chr(ord("A") + idx) for idx in range(self.num_actions())]
        if mask_indices is not None:
            tokens = [
                token for idx, token in enumerate(tokens) if idx not in mask_indices
            ]
        return tokens

    def action_token_to_index(self, action_token):
        """
        Converts an action token (like 'A', 'B', etc.) to its corresponding action index.

        :param action_token: String representing the action token.
        :return: Integer index of the action.
        """
        return ord(action_token) - ord("A")

    def token_description(self):
        return "english uppercase letter"

    def get_action_index(self, action_str):
        """
        Retrieves the index of an action given its name.

        :param action_str: String name of the action.
        :return: Integer index of the action.
        """
        for action_name, action_info in self.action_map.items():
            if action_name == action_str:
                return action_info["index"]
        raise ValueError(f"Invalid action string: {action_str}")

    def get_action_name(self, action_index):
        for action_name, action_info in self.action_map.items():
            if action_info["index"] == action_index:
                return action_name
        raise ValueError(f"Invalid action index: {action_index}")


if __name__ == "__main__":

    env_kwargs = {
        "area": (64, 64),  # world have 64x64 elems, this is the grid
        "view": (9, 9),  # the player sees 9x9 elems from the grid. 9x9 is default
        # "view": (
        #     6,
        #     6,
        # ),  # * we can make it 6x6 to see 36 elems so we can caption all to 1 prompt,
        # and allow LLM to be better
        "size": (
            64,
            64,
        ),  # resolution of the image for the obs, if make it bigger: see the same view elems in better resolution.
        "reward": True,
        "length": 10000,
        "seed": None,
    }

    env = crafter.Env(**env_kwargs)  # Create the environment
    obs = env.reset()  # Start a new environment
    _, _, _, state_info = env.step(0)  # Take an action (example: action index 0)

    inventory, achievements, semantic, player_pos, reward = (
        state_info["inventory"],
        state_info["achievements"],
        state_info["semantic"],
        state_info["player_pos"],
        state_info["reward"],
    )

    action_captioner = ActionCaptioner(constants.crafter_actions_map)
    print(action_captioner.caption_all())

    # # To be sure our mapping is like the real actions:
    # real_action_map = env.action_names
    # # Print action mapping
    # for idx, name in enumerate(real_action_map):
    #     print(f"{idx}: {name}")

    # state_captioner = StateCaptioner(env)
    # # state_captioner = RelativeGridCaptioner(env)

    # size_around_player_to_caption = None
    # # size_around_player_to_caption = "all"  # all the area

    # mask_objects_list = ["grass"]
    # state_description = state_captioner.caption(
    #     state_info,
    #     mask_objects_list=mask_objects_list,
    #     size_around_player_to_caption=size_around_player_to_caption,
    # )
    # # state_description = state_captioner.caption(semantic, player_pos)
    # print(state_description)

    # move_right_index = action_captioner.get_action_index("move_right")
    # move_right_str = action_captioner.caption(move_right_index)

    # move_left_index = action_captioner.get_action_index("move_left")
    # move_left_str = action_captioner.caption(move_left_index)
    # last_transitions = [
    #     {
    #         "state_caption": "Player near a zombie, low on health.",
    #         "skill_used": "Skill: run away from monster",
    #         "action_taken": f"Action: {move_left_index}: {move_left_str}",
    #         "reward": -2,
    #     },
    #     {
    #         "state_caption": "Player still near the zombie.",
    #         "skill_used": "Skill: run away from monster",
    #         "action_taken": f"Action: {move_left_index}: {move_left_str}",
    #         "reward": -1,
    #     },
    #     {
    #         "state_caption": "Player gained distance, zombie is farther.",
    #         "action_taken": f"Action: {move_right_index}: {move_right_str}",
    #         "reward": 0,
    #     },
    # ]

    # skill_policy_desc = (
    #     "Policy: Run away from monsters. If stuck in front water - try another path."
    # )

    # prompt_generator = prompts_builder.PromptBuilderForActionSelection(
    #     skill_policy_desc, action_captioner, world_info
    # )

    # prompt = prompt_generator.generate_prompt(state_description, last_transitions)
    # print(prompt)

    # img = env.render()  # to see as image
    # print(f"img shape: {obs.shape}")  # (64, 64, 3)

    # import matplotlib.pyplot as plt

    # plt.imshow(img)
    # plt.show()  # not alwys see all the 9x9 view..

    # action_space = env.action_space
    # print(action_space)  # Discrete(17)
    # action = action_space.sample()
    # obs, reward, done, info = env.step(action)
    # #     info = {
    # #     'inventory': self._player.inventory.copy(),
    # #     'achievements': self._player.achievements.copy(),
    # #     'discount': 1 - float(dead),
    # #     'semantic': self._sem_view(),
    # #     'player_pos': self._player.pos,
    # #     'reward': reward,
    # # }

    # print(obs.shape)
    # import matplotlib.pyplot as plt

    # plt.imshow(obs)
    # plt.show()

    # print(info)

    # sem = info["semantic"]
    # sem.shape  # (64, 64)# 2D ids array, as the area shape, it gives info about all the grid and not only the observed domain.
