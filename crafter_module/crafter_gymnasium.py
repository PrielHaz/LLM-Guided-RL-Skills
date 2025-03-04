# * Crafter Captioner take as argument the self.raw_crafter and not this env class.
import os
import sys
from typing import List, Optional

try:  # if py file
    sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
except:  # if ipynb file will run this
    sys.path.append(os.path.abspath(".."))

# use importlib.reload(module)
import importlib

import crafter
import gymnasium as gym
import numpy as np

from crafter_module.crafter_captioner import ActionCaptioner, StateCaptioner
from nlp_module.prompts_builder import PromptBuilderForActionSelection
from utils import constants, util_funcs
from utils.llm_skills_args import LLM_Skills_Args, Skill


# Need all args to be kwargs for simplicity in creating env duplications for eval\train
class CrafterGymnasium(gym.Env):
    def __init__(
        self,
        raw_env_kwargs=None,
        llm_skills_args: Optional[LLM_Skills_Args] = None,
        verbose=0,
        print_file_path=None,
        mask_actions_indices=[],
    ):
        """
        Custom Crafter environment wrapped for Gymnasium, with optional LLM-based skill augmentation.

        Args:
            raw_env_kwargs (dict): Arguments for creating the Crafter environment.
            llm_skills_args (Optional[LLM_Skills_Args]): If provided, enables skill-based actions.
        """
        super().__init__()
        assert raw_env_kwargs is not None, "raw_env_kwargs must be provided."
        self.raw_env_kwargs = raw_env_kwargs
        self.raw_env = crafter.Env(**raw_env_kwargs)

        # Initialize captioners using the raw env
        self.action_captioner = ActionCaptioner(constants.crafter_actions_map)
        self.masked_action_tokens = self.action_captioner.get_actions_tokens_list(
            mask_indices=mask_actions_indices
        )  # like ['B', 'C'] for 2 actions where A masked
        self.state_captioner = StateCaptioner(self.raw_env)
        self.prompt_builder = PromptBuilderForActionSelection(
            self.action_captioner, constants.crafter_world_info
        )

        # Define observation and action spaces
        self.observation_space = gym.spaces.Box(
            0,
            255,
            self.raw_env.observation_space.shape,
            dtype=self.raw_env.observation_space.dtype,
        )

        self.llm_skills_args = llm_skills_args
        self.using_llm_skills = llm_skills_args is not None
        self.num_skills = None
        # Expand action space if LLM skills are provided
        if self.using_llm_skills:
            self.num_skills = len(llm_skills_args.skills)
            self.action_space = gym.spaces.Discrete(
                self.raw_env.action_space.n
                + self.num_skills  # add the option skills as actions
            )

        else:
            self.action_space = gym.spaces.Discrete(self.raw_env.action_space.n)

        # Store recent transitions for prompt generation
        self.recent_transitions: List[dict] = []
        self.last_state_caption = None
        self.mask_actions_indices = mask_actions_indices
        # it includes if reset env stays the same
        self.total_steps_counter = 0
        # prints:
        self.verbose = verbose
        self.print_file_path = print_file_path
        self.logger = util_funcs.logger(verbose, print_file_path)

    def reset(self, seed=None):
        # old:
        # self.raw_env_kwargs["seed"] = seed
        # self.raw_env = crafter.Env(**self.raw_env_kwargs)

        # * The reset of raw_env uses the raw_env seed to generate the
        # world, so let's recreate the raw_env with the seed passed here.
        # But if we do so it might harm things.. The best way
        # is to init the env seed as they do in the crafter code.
        # They use seed only in reset to create the world so we will
        # change the seed and then reset the raw_env:
        # create seed as in crafter.Env init:
        old_seed = self.raw_env._seed
        self.raw_env._seed = np.random.randint(0, 2**31 - 1) if seed is None else seed
        # now the creation of the world will use the new seed:
        obs = self.raw_env.reset()
        self.raw_env._seed = old_seed  # restore the original seed.
        info = {
            # here we need to add info like sum of int_rewards so we know the true rewards of the game excluding ours and things like this.
            # also maybe info like if skill used etc. Or maybe just track it and write to some files so we can access later
        }
        # Reset the recent transitions and last state caption
        self.recent_transitions = []
        self.last_state_caption = None
        return obs, info  # gymnasium returns info too.

    def step(self, action):
        # util_funcs.log(
        #     f"\n\n&&&&&&& Total steps made in env: {self.total_steps_counter} &&&&&&&", level
        # )
        self.total_steps_counter += 1
        self.logger.log(
            f"\n\n&&&&&&& Step number executing in env: {self.total_steps_counter} &&&&&&&",
            level=1,
        )
        transition = {
            "state_caption": None,
            "skill_used": None,
            "action_taken": None,
            "reward": None,
        }
        chosen_action_index = (
            action  # start with the original action chosen by the agent.
        )
        if (
            self.last_state_caption is None
        ) and chosen_action_index >= self.raw_env.action_space.n:
            self.logger.log(
                f"@ Agent chose first action as a skill, translating to defualt action: {self.action_captioner.caption(self.llm_skills_args.default_action_index)}",
                level=1,
            )
            # No transitions yet(first action in env) and action is a skill, LLM cannot be used, choose defualt.
            # it will enter the primitive if below that not need transitions.
            chosen_action_index = self.llm_skills_args.default_action_index

        # If action is a primitive action, execute it directly
        if (
            not self.using_llm_skills
            or chosen_action_index < self.raw_env.action_space.n
        ):
            self.logger.log(
                f"# Agent chose primitive action: {self.action_captioner.caption(chosen_action_index)}",
                level=1,
            )

            # raw env return: (obs, reward, done, info)
            obs, reward, done, info = self.raw_env.step(chosen_action_index)
        else:
            # Now we know for sure that self.last_state_caption is not None so can gen prompt
            # Select an LLM-generated action
            skill_index = chosen_action_index - self.raw_env.action_space.n
            skill_desc = self.llm_skills_args.skills[skill_index].description
            skill_name = self.llm_skills_args.skills[skill_index].name

            messages = self.prompt_builder.generate_prompt(
                skill_desc,
                self.last_state_caption,
                self.recent_transitions[
                    :-1
                ],  # the last elem is the last state, we need only past events here.
                mask_actions_indices=self.mask_actions_indices,
            )
            self.logger.log(
                f"$ Agent chose index {action} which translates to skill index: {skill_index}"
            )
            self.logger.log(
                f"Skill name: {skill_name}",
            )
            self.logger.log(f"Skill description: {skill_desc}", level=2)
            self.logger.log(f"\n\nRun option policy...")
            chosen_action_index = self.llm_skills_args.option_policy.predict(
                messages,
                self.action_captioner,
                self.masked_action_tokens,
                default_action_index=self.llm_skills_args.default_action_index,
            )
            # assert its a primitive action:
            assert (
                chosen_action_index < self.raw_env.action_space.n
                and chosen_action_index >= 0
            ), f"Option Policy returned an invalid action index(out of bounds): {chosen_action_index}."

            # Execute the predicted action
            obs, reward, done, info = self.raw_env.step(chosen_action_index)
            transition["skill_used"] = skill_name
            self.logger.log(
                f"## Option policy chose primitive action: {self.action_captioner.caption(chosen_action_index)}"
            )
        if self.using_llm_skills:
            transition["action_taken"] = self.action_captioner.caption(
                chosen_action_index
            )
            transition["reward"] = reward
            transition["state_caption"] = self.state_captioner.caption(
                info,
                explain_grid_format=False,
                explain_achieved=False,
                explain_not_achieved=False,
            )  # transitions we store we dont want grid explanations in them.
            self.last_state_caption = self.state_captioner.caption(
                info,
                explain_grid_format=True,
            )  # we do want explanations in the state caption we give next to LLM
            self.recent_transitions.append(transition)
            # Keep only the last `num_steps_pass_llm` + 1 transitions. Even if list has 1 or 0 elements it works as expected and remain them.
            # The +1 is for the current state.
            self.recent_transitions = self.recent_transitions[
                -self.llm_skills_args.num_steps_pass_llm - 1 :
            ]
        self.logger.log(
            f"** Reward: {reward}, Done: {done}",
        )
        return (
            obs,
            reward,
            done,
            False,
            info,
        )  # Gymnasium requires the truncated flag so we return False.

    def render(self, size=None):
        return self.raw_env.render(size)

    def close(self):
        pass
