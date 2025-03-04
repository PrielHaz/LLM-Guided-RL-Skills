import os
import pathlib
import sys
from typing import List, Optional

import transformers
from openai import OpenAI

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))


from abc import ABC, abstractmethod
from typing import Dict, List, Optional

# this line for the use of genai.GenerativeModel
import google.generativeai as google_genai
import torch

# This line for the use of gemini api:
from google import genai
from google.genai import types
from termcolor import colored
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline

from crafter_module import crafter_captioner
from nlp_module import prompts_builder
from utils import constants, util_funcs


class OptionPolicyResponse:
    def __init__(self, most_prob_token, probs_for_unmasked):
        """
        Response class for the OptionPolicy classifier.
        :param most_prob_token: The most probable action token.
        :param probs_for_unmasked: A list of probabilities (logits) for all unmasked tokens.
        """
        self.most_prob_token = most_prob_token
        self.probs_for_unmasked = probs_for_unmasked

    def __str__(self):
        return f"Response(most_prob_token={self.most_prob_token}, \nprobs_for_unmasked={self.probs_for_unmasked})"


class OptionPolicy(ABC):
    @abstractmethod
    def classify(
        self, messages, masked_action_tokens: List[str]
    ) -> OptionPolicyResponse:
        pass

    def predict(
        self,
        messages,
        action_captioner,
        masked_action_tokens: List[str],
        default_action_index: int = 5,
    ) -> int:
        """
        Predicts the index of the action with the highest probability.

        Args:
            messages: The messages list of dictionaries with 'role' and 'content' keys
            masked_action_tokens (List[str]): List of available action tokens
            default_action_index (int): Default action index if no valid action is chosen

        Returns:
            int: Index of the selected action
        """
        response = self.classify(messages, masked_action_tokens)
        if response.most_prob_token in masked_action_tokens:
            # the unmasked action tokens not have the correct index so need this:
            return action_captioner.action_token_to_index(response.most_prob_token)

        # Using default action:
        # if self has attribute verbose and it is True, print the default action index
        if hasattr(self, "verbose") and self.verbose > 0:
            print(
                f"No valid action token found. Using default action index: {default_action_index}"
            )

        return default_action_index


class DeepseekClassifier(OptionPolicy):
    def __init__(self, model="deepseek-chat", verbose=0, print_file_path=None):
        """
        Initialize the DeepseekClassifier with the specified model.

        Args:
            model (str): The Deepseek model to use
            verbose (bool): Whether to print debug information
            chat_tokenizer_dir (str): Path to the tokenizer directory, if None will use default path
        """
        self.client = OpenAI(
            base_url="https://api.deepseek.com",  # Deepseek API endpoint
            api_key=os.getenv("DEEPSEEK_API_KEY"),  # Get API key from environment
        )
        self.model = model
        self.verbose = verbose
        self.print_file_path = print_file_path
        self.logger = util_funcs.logger(verbose, print_file_path)

    def classify(
        self, messages, masked_action_tokens: List[str]
    ) -> OptionPolicyResponse:
        """
        Classifies the context and returns probabilities for each action token.

        Args:
            context (str): The context string containing situation and policy description
            masked_action_tokens (List[str]): List of available action tokens

        Returns:
            OptionPolicyResponse: Contains the most probable token and probability list
        """
        self.logger.log(
            f"Input for Deepseek:\n\n{util_funcs.get_prompt_messages_nicely(messages)}",
            level=2,
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=1.0,  # not matter since we take direct logprobs
            max_tokens=1,  # We only need one token
            logprobs=True,  # Return log probabilities
            top_logprobs=20,  # Get top 20 tokens
            n=1,  # Single completion
            seed=0,  # For reproducibility
            stream=False,
        )
        raw_response_txt = response.choices[0].message.content
        # CompletionUsage(completion_tokens=47, prompt_tokens=6, total_tokens=53, completion_tokens_details=None, prompt_tokens_details=PromptTokensDetails(audio_tokens=None, cached_tokens=0), prompt_cache_hit_tokens=0, prompt_cache_miss_tokens=6)
        usage = response.usage
        self.logger.log(
            f"Deepseek {self.model} raw response: {raw_response_txt}", level=1
        )
        self.logger.log(
            f"Usage: prompt tokens= {usage.prompt_tokens}, completion tokens= {usage.completion_tokens}, total tokens= {usage.total_tokens}, prompt_cache_hit_tokens= {usage.prompt_cache_hit_tokens}, prompt_cache_miss_tokens= {usage.prompt_cache_miss_tokens}",
            level=1,
        )
        # Extract the choice (should be only one)
        choices = response.choices
        assert len(choices) == 1, "Expected only one choice"
        choice_logprobs = choices[0].logprobs

        # Check if there was a refusal from the model
        assert choice_logprobs.refusal is None, "Model refused to answer"

        # Process the log probabilities
        probs_for_unmasked = [None] * len(masked_action_tokens)
        most_prob_token = None

        # self.logger.log("Top logprobs from model response:", level=1)

        for prob in choice_logprobs.content[0].top_logprobs:
            token = prob.token
            logprob = prob.logprob

            # self.logger.log(f"  Token: {token}, LogProb: {logprob}", level=1)

            if token in masked_action_tokens:
                index = masked_action_tokens.index(token)
                probs_for_unmasked[index] = logprob

                # Update most_prob_token if this is the first valid token found
                # or if it has a higher probability than the current most_prob_token
                if (
                    most_prob_token is None
                    or logprob
                    > probs_for_unmasked[masked_action_tokens.index(most_prob_token)]
                ):
                    most_prob_token = token

        self.logger.log(f"Most probable token: {most_prob_token}", level=1)

        return OptionPolicyResponse(most_prob_token, probs_for_unmasked)


# The 7B not works good at all!
# class DeepSeekLocalClassifier(OptionPolicy):
#     def __init__(self, model_name="deepseek-ai/deepseek-llm-7b-chat"):
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.model = AutoModelForCausalLM.from_pretrained(
#             model_name, torch_dtype=torch.bfloat16, device_map="auto"
#         )
#         self.tokenizer.pad_token = self.tokenizer.eos_token
#         self.model.generation_config = GenerationConfig.from_pretrained(model_name)
#         self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

#     def classify(
#         self, messages, masked_action_tokens: List[str]
#     ) -> OptionPolicyResponse:
#         input_tensor = self.tokenizer.apply_chat_template(
#             messages,
#             add_generation_prompt=True,
#             return_tensors="pt",
#         )
#         # ne(not equal) checks elem wise if not equal to pad token id.
#         # attention_mask is 1 for real tokens and 0 for padding tokens.
#         attention_mask = input_tensor.ne(self.tokenizer.pad_token_id).long()

#         with torch.no_grad():
#             outputs = self.model.generate(
#                 input_tensor.to(self.model.device),
#                 attention_mask=attention_mask.to(self.model.device),
#                 max_new_tokens=1,
#                 return_dict_in_generate=True,
#                 output_scores=True,
#                 return_legacy_cache=True,
#             )
#         next_token_logits = outputs.scores[0][0]
#         probabilities = torch.nn.functional.softmax(next_token_logits, dim=-1)
#         # TODO probably we neet iterate from high to low prob and if in masked_action_tokens then take it.
#         top_20_probs, top_20_indices = torch.topk(probabilities, 20)
#         top_20_tokens = self.tokenizer.convert_ids_to_tokens(top_20_indices.tolist())
#         probs_for_unmasked = [None] * len(masked_action_tokens)
#         most_prob_token = None
#         for token, prob in zip(top_20_tokens, top_20_probs.tolist()):
#             if token in masked_action_tokens:
#                 index = masked_action_tokens.index(token)
#                 probs_for_unmasked[index] = prob
#                 if most_prob_token is None:
#                     most_prob_token = token

#         # To print top20 tokens:
#         # print("\nTop 20 Next Token Predictions:")
#         # for token, prob in zip(top_20_tokens, top_20_probs.tolist()):
#         # print(f"{token}: {prob:.2f}")
#         return OptionPolicyResponse(most_prob_token, probs_for_unmasked)


# * For gemini to work need translate messages to context.
# class GeminiClassifier(OptionPolicy):
#     def __init__(self, model_name="gemini-2.0-flash", verbose=0):
#         api_key = os.getenv("GEMINI_API_KEY")
#         self.client = genai.Client(api_key=api_key)
#         self.model_name = model_name
#         self.model = google_genai.GenerativeModel(self.model_name)
#         self.verbose = verbose

#     def count_tokens(self, context: str) -> int:
#         return self.model.count_tokens(context)

#     def classify(
#         self, messages, masked_action_tokens: List[str]
#     ) -> OptionPolicyResponse:
#         print context to model in purple:
#         if self.verbose:
#             print(
#                 colored(
#                     f"Context(num_tokens={self.count_tokens(context)}): {context}",
#                     "magenta",
#                 )
#             )
#         response = self.client.models.generate_content(
#             model=self.model_name,
#             contents=context,
#             config=types.GenerateContentConfig(
#                 max_output_tokens=1,
#                 temperature=0.01,  # Low temp for deterministic behavior
#             ),
#         )
#         response_text = response.text.strip()
#         if self.verbose:
#             print(f"% Gemini Response(len={len(response_text)}): {response_text}")
#         probs_for_unmasked = [0.0] * len(masked_action_tokens)
#         most_prob_token = None
#         if response_text in masked_action_tokens:
#             index = masked_action_tokens.index(response_text)
#             probs_for_unmasked[index] = 1.0
#             most_prob_token = response_text
#             if self.verbose:
#                 print(f"Response is token index: {index}")

#         return OptionPolicyResponse(most_prob_token, probs_for_unmasked)


# Basic Testing:
if __name__ == "__main__":
    action_captioner = crafter_captioner.ActionCaptioner(constants.crafter_actions_map)
    mask_actions_indices = [0]  # noop
    masked_actions_tokens_lst = action_captioner.get_actions_tokens_list(
        mask_indices=mask_actions_indices
    )
    print(f"Masked actions tokens list: {masked_actions_tokens_lst}")

    # Initialize the classifier with GPU/CPU selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # option_policy = DeepSeekClassifier()
    # option_policy = GPTClassifier()
    option_policy = DeepseekClassifier(verbose=1)

    # option_policy_desc = "Drinking water skill. Policy -> always go for the water, choose action do when you are in front of water."
    skill0_txt_path = "../skills_gen/skills_created_sessions/deepseekV3/option_policies_descriptions/skill0.txt"
    with open(skill0_txt_path, "r") as f:
        option_policy_desc = f.read()

    world_info = constants.crafter_world_info

    # world_info = "crafter world"
    prompt_builder_for_action_selection = (
        prompts_builder.PromptBuilderForActionSelection(
            action_captioner,
            world_info,
        )
    )
    cur_state_caption = "2 steps to the left there is water. 2 steps to the right there is wood you can collect. Up there is a monster."
    messages = prompt_builder_for_action_selection.generate_prompt(
        option_policy_desc, cur_state_caption, mask_actions_indices=mask_actions_indices
    )
    print(f"\n\messages:@@@\n{messages}\n@@@")
    response = option_policy.classify(messages, masked_actions_tokens_lst)
    print(f"\n\nResponse: {response}")
