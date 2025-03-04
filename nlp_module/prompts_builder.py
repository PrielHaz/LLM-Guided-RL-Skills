import random


class PromptBuilderForActionSelection:
    """Generates a structured prompt for LLM skill selection based on world info, state, and action history."""

    def __init__(self, action_captioner, world_info):
        """
        Initializes the GetPrompt class.

        Args:
            action_captioner (ActionCaptioner): Provides mappings for actions.
            world_info (str): General information about the game world.
        """
        self.action_captioner = action_captioner
        self.world_info = world_info

    def generate_prompt(
        self,
        option_policy_desc,
        cur_state_caption,
        last_transitions=None,
        mask_actions_indices=[],
        messages_format=True,  # If want input like deepseek\GPT.
    ):
        """
        Generates a structured prompt for the LLM to select an action.

        Args:
            last_transitions (list of dict): Each dict contains 'state_caption', 'action_taken', and 'reward'.
            messages_format (bool): If True, returns the prompt as a list of messages instead of a single string.

        Returns:
            str or list: The complete formatted prompt as a string or a list of messages.
        """
        masked_actions_tokens = self.action_captioner.get_actions_tokens_list(
            mask_indices=mask_actions_indices
        )

        system_msg = "You are an intelligent agent in a game environment."

        game_info_msg = f"Game true information:(((\n{self.world_info}\n)))\n"

        actions_msg = "Available primitive actions:\n"
        actions_msg += self.action_captioner.caption_all(
            mask_indices=mask_actions_indices
        )

        example_msg = f"""
Example toy Question-Answer (Follow this format only always!):
Question: [[[\nYou follow option policy: chop trees. You see tree in front of you. What should be the next action?\n]]]
Answer: """
        example_answer_msg = "F"
        post_example_answer_msg = f"In this example, the next action should be F which relates to 'do' that chops the tree in front of you."

        history_msg = ""
        if last_transitions:  # if not None, not empty list etc...
            history_msg += "Recent game events from the old to the most recent:\n"
            for transition_i, transition in enumerate(last_transitions):
                was_before = len(last_transitions) - transition_i
                history_msg += (
                    f"State before {was_before} steps:\n{transition['state_caption']}\n"
                )
                if "skill_used" in transition and transition["skill_used"] is not None:
                    history_msg += (
                        f"Skill used: {transition['skill_used']} which resulted in:\n"
                    )
                history_msg += f"Primitive Action taken: {transition['action_taken']}\n"
                history_msg += f"Reward received: {transition['reward']}\n"
        history_add = (
            " to the recent game events mentioned above and" if history_msg else ""
        )
        user_query = f"""
Question: [
The current state is:\n{cur_state_caption}\n\n
You need to choose the action that fits the current situation, considering{history_add} to this option policy description:\n[{option_policy_desc}]\n
The option policy outlines the general goals of the option policy and not what to do in all situations, just some of them.
So think what needs to be done in order to achieve the goals of the option policy and choose adequate action, you need to think for long term.
For example, if the option policy is to chop trees, locate a near tree in the grid, and choose a move action that brings you closer to the tree.
When the player facing the tree, only then you can choose 'do' action that chops the tree.

Write only 1 {self.action_captioner.token_description()} from those:
{", ".join(masked_actions_tokens)}
(see the primitive actions above).
]
Answer: """

        # * To allow deepseek context caching see: https://api-docs.deepseek.com/guides/kv_cache
        # We give first the repeated messages, then the messages that change across calls to API
        # to minimize cost.
        if messages_format:
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": game_info_msg},
                {"role": "user", "content": actions_msg},
                {"role": "user", "content": example_msg},
                {"role": "assistant", "content": example_answer_msg},
                {"role": "user", "content": post_example_answer_msg},
            ]
            if history_msg:
                messages.append({"role": "user", "content": history_msg})
            messages.append({"role": "user", "content": user_query})
            return messages

        # Default: return as a single string prompt
        msgs_ordered = [
            system_msg,
            game_info_msg,
        ]
        # If not caching, history better fit this place:
        if history_msg:
            msgs_ordered.append(history_msg)
        msgs_ordered.extend(
            [
                actions_msg,
                example_msg,
                example_answer_msg,
                post_example_answer_msg,
                user_query,
            ]
        )
        return "\n".join(msgs_ordered)
