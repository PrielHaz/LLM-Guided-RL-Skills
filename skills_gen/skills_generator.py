import os
import sys

import cv2
import imageio

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
import argparse
from abc import ABC, abstractmethod
from typing import Dict, List

import crafter
import numpy as np
import openai
import stable_baselines3
import torch
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer

from crafter_module.crafter_captioner import ActionCaptioner, StateCaptioner
from nlp_module.prompts_builder import PromptBuilderForActionSelection
from utils import constants, util_funcs
from utils.constants import crafter_actions_map, crafter_world_info


class SkillsGenerator(ABC):
    """
    Abstract class for skill generation in an RL environment.
    """

    @abstractmethod
    def generate_option_policy(self, context: str) -> str:
        """
        Generate a new skill policy based on the given context.
        """
        pass


# class GPTSkillsGenerator(SkillsGenerator):
#     """
#     ChatGPT-based skills generator.
#     """

#     def __init__(self, model="gpt-4-turbo"):
#         self.model = model

#     def generate_option_policy(self, context: str) -> str:
#         response = openai.ChatCompletion.create(
#             model=self.model,
#             messages=[{"role": "user", "content": context}],
#             temperature=0.7,
#             max_tokens=500,
#         )
#         return response["choices"][0]["message"]["content"].strip()


class DeepseekSkillsGenerator:
    """
    Deepseek-based skills generator using the OpenAI-compatible API.
    """

    def __init__(self, model="deepseek-chat"):
        """
        Initialize the DeepseekSkillsGenerator with the specified model.

        Args:
            model (str): The Deepseek model to use, defaults to "deepseek-chat"
        """
        self.client = OpenAI(
            base_url="https://api.deepseek.com",  # Deepseek API endpoint
            api_key=os.getenv("DEEPSEEK_API_KEY"),  # Get API key from environment
        )
        self.model = model

    def generate_option_policy(self, context: str) -> str:
        """
        Generate a skill policy based on the given context.

        Args:
            context (str): The context for skill generation

        Returns:
            str: Generated skill policy description
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": context}],
            temperature=0.7,
            n=1,
            stream=False,
        )

        raw_response_txt = response.choices[0].message.content
        return raw_response_txt.strip()


class DeepSeekLocalSkillsGenerator(SkillsGenerator):
    """
    DeepSeek-based skills generator, using a locally hosted model.
    """

    def __init__(self, model_name="deepseek-ai/deepseek-llm-7b-chat"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_option_policy(self, context: str) -> str:
        input_tensor = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": context}],
            add_generation_prompt=True,
            return_tensors="pt",
        )
        attention_mask = input_tensor.ne(self.tokenizer.pad_token_id).long()

        with torch.no_grad():
            outputs = self.model.generate(
                input_tensor.to(self.model.device),
                attention_mask=attention_mask.to(self.model.device),
                max_new_tokens=100,
            )
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text.strip()


def generate_skills(
    num_skills: int,
    skills_generator: SkillsGenerator,
    model,
    env,
    use_transfer_learning: bool = False,
    traj_len=100,
    existing_skills: List[str] = None,
    action_captioner: ActionCaptioner = None,
    mask_actions_indices: List[int] = [],
):
    """
    Generate skills from RL model using LLMs.
    """
    captioner = StateCaptioner(env)

    option_policies = [] if existing_skills is None else existing_skills.copy()
    os.makedirs("captioned_trajectories", exist_ok=True)
    os.makedirs("option_policies_descriptions", exist_ok=True)
    os.makedirs("context_for_skill_generation", exist_ok=True)

    print(
        f"Already generated skills: {len(option_policies)}, complete to {num_skills}:"
    )
    for i in range(len(option_policies), num_skills):
        print(f"Generating skill {i}...")
        obs = env.reset()
        traj_caption = []
        short_traj_text = ""
        frames = []  # Store frames for video
        actions_got_positive_reward_str = []

        # Set up video parameters
        fps = 10  # Frames per second
        video_path = f"./captioned_trajectories/traj{i}.mp4"
        for step_i in range(traj_len):
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            explain_grid_format = False
            if step_i == 0:
                explain_grid_format = True
            state_desc = captioner.caption(
                info,
                explain_grid_format=explain_grid_format,
                explain_achieved=True,
                explain_not_achieved=True,
            )
            action_desc = action_captioner.caption(action)
            transition_caption = f"$ Current State:[[{state_desc}]]\n\n\n@ Action took: [[{action_desc}]]\n\n\n& Reward: {reward}\n\n"
            traj_caption.append(f"Step {len(traj_caption) + 1}:\n{transition_caption}")

            action_text = f"Act:{action_captioner.get_action_name(action)}"
            reward_text = f"Rew:{reward:.2f}"
            if reward > 0:
                actions_got_positive_reward_str.append(
                    f"Step {len(traj_caption)}: {action_text}({reward_text})"
                )

            short_traj_text += (
                f"S{len(traj_caption)}: {action_text:20}({reward_text})\n"
            )
            annotated_obs = util_funcs.annotate_obs(obs, action_text, reward_text)
            frames.append(annotated_obs)
            if done:
                break

        traj_text = "\n".join(traj_caption)
        with open(f"captioned_trajectories/traj{i}.txt", "w") as f:
            f.write(traj_text)
        with open(f"captioned_trajectories/short_traj{i}.txt", "w") as f:
            f.write(short_traj_text)

        # save the video of the trajectory to captioned_trajectories dir:
        imageio.mimsave(video_path, frames, fps=fps)
        print(f"Saved trajectory video: {video_path}")
        context = f"Trajectory:\n{traj_text}\n\n"

        context += (
            f"\n\nWorld information:{constants.crafter_world_info}\nEnd world info.\n"
        )
        context += f"\n\nPrimitive actions:\n{action_captioner.caption_all(mask_indices=mask_actions_indices)}\nEnd of primitive actions.\n\n"

        if not option_policies:
            context += f"No skills discovered yet! Can propse any new skill.\n"
        else:
            context += f"Previously discovered skills(don't propose these):\n"
            for j, skill in enumerate(option_policies):
                context += f"Skill policy {j}:\n{skill}\n\n\n"
        context += f"End of previously discovered skills.\n\n"

        context += f"Actions that got positive rewards:\n"
        for action_str in actions_got_positive_reward_str:
            context += f"{action_str}\n"

        # Optional text for transfer learning:
        # We aim to leverage this agent's skills descriptions to enhance the training of new agents.
        #  based on the skills the current agent employs to earn rewards

        context += f"""Based on world information and environmental interactions, generate a new skill (one that is not similar to any we have already discovered) we cab use to maximize rewards and accomplish various achievements in the environment.
You need to describe a skill of the agent, as observed in its trajectory, that the agent uses to gain rewards.
"""
        if use_transfer_learning:
            context += f"""Transfer learning: We aim to leverage this agent's skills descriptions to enhance the training of new agents.
Base your proposed skill on the skills the current agent employs to earn rewards.
"""
            context += f"""
Format your response as follows:

LLM Answer: Option skill 'Search for water'. This policy directs the agent to go to nearby water and execute the 'do' action whenever water is detected in front of the player.
Write more goals here...
If the situation not fits to the scenarios below-think how to achieve these goals in long term and select best action.
Scenarios sorted by importance(1 is most important):
1. If water is observeable and drink<=4: Move towards water, if in front of you: select 'do' action.
2. ... for all scenarios.

Now, suggest a new option policy(detailed, not like the example above).
Explain what it should do in all scenarios sorted by importance. For example, say in the cell:
Select and execute the goal that seems the most relevant according to the situation:
1. If cow is observeable and food <= 4 and eat_cow achievement still not achieved: Move towards cow, if in front of you: select 'do' action.
2. If tree is nearby and ....
3. If stone is observeable and ...
...
for all the possible scenarios the agent can encounter in the environment.
Write the scenarios in order of importance for this specific option policy! If it's food based policy, the cow rule can be first for example.
But for each option policy write the scenarios in order of importance for that specific policy.
Remember to instruct the agent to move towards the object before telling it to perform an action on it(like 'do'), so that the agent doesn't get stuck.
Also, write 5-7 sentences on the general goals of the option policy.
After that, write that if the situation not fits to the scenarios below-think how to achieve these goals in long term and select best action.
Write in raw text in your black cell.

LLM Answer: """

        # print(f"#### Context for skill {i+1}:\n{context}\n")
        with open(f"context_for_skill_generation/context{i}.txt", "w") as f:
            f.write(context)

        # * We can take context by context and enter it to GPT website and then copy paster the response skill.
        new_skill = skills_generator.generate_option_policy(context)
        option_policies.append(new_skill)

        with open(f"option_policies_descriptions/skill{i}.txt", "w") as f:
            f.write(new_skill)

    return option_policies


# * Run inside skills_gen directory: python skills_generator.py
# Add: --from_non_trained  if want to generate from non-trained agent.
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Skill Generation in Crafter Environment"
    )

    # Add an optional flag for using non-trained agent
    parser.add_argument(
        "--from_non_trained",
        action="store_true",
        default=False,
        help="Use a non-trained agent with random weights",
    )
    args = parser.parse_args()

    # Determine model path and transfer learning based on flag
    if args.from_non_trained:
        model_path = "./non_trained_agent.zip"
        use_transfer_learning = (
            False  # non trained so we only use trajectories to generate skills
        )

        print("\n🤖 Non-Trained Agent Mode 🤖")
        print("-------------------------------")
        print("Using a non-trained agent with random initial weights.")
        print("This means skills will be generated from pure exploration.")
        print("No prior learning transfer will be applied.\n")
    else:
        model_path = "./1M_trained_agent.zip"  # this is the model in "../results/crafter/primitive/1M/final_model.zip"
        use_transfer_learning = True  # transferring from trained agent

        print("\n🧠 Trained Agent Mode 🧠")
        print("-------------------------------")
        print("Using a pre-trained agent with 1M training steps.")
        print("Skills will be generated leveraging the agent's learned behaviors.")
        print("Transfer learning is enabled to extract meaningful policy insights.\n")

    action_captioner = ActionCaptioner(crafter_actions_map)
    existing_skills = []
    # load option_policies_descriptions all txt files there to existing_skills list:
    if os.path.exists("option_policies_descriptions"):
        for file in os.listdir("option_policies_descriptions"):
            with open(os.path.join("option_policies_descriptions", file), "r") as f:
                existing_skills.append(f.read())
    # Maximum 9 skills: 17 primitive + 9 generated = 26, so it will be exactly A-Z actions.
    # num_skills = 8
    num_skills = 4

    # Can use len+1, each time run LLM with generated context and add the skill to the list and run again.
    # num_skills = len(existing_skills) + 1
    traj_len = 20

    print(f"Existing Skills:(len={len(existing_skills)})\n{existing_skills}")

    env = crafter.Env()
    model = stable_baselines3.PPO.load(model_path, env=env)

    # skills_generator = GPTSkillsGenerator()
    skills_generator = DeepseekSkillsGenerator()
    mask_actions_indices = [0]  # noop, we want option policies to be done and not noops
    skills = generate_skills(
        num_skills,
        skills_generator,
        model,
        env,
        use_transfer_learning=use_transfer_learning,
        traj_len=traj_len,
        existing_skills=existing_skills,
        action_captioner=action_captioner,
        mask_actions_indices=mask_actions_indices,
    )
    print("Generated Skills:", skills)
