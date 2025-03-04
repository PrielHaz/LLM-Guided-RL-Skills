## First, define in your environment

export DEEPSEEK_API_KEY="Here your API key"

## Packages:

run: pip install crafter
Other packages are standard RL and python packages.

## Generating skills Descriptions:

Run from the project root:
cd skills_gen

Then, to generate skills from primitive agent's trajectories trained for 1M steps run:
python skills_generator.py

To generate skills from primitive agent with random weights use:
python skills_generator.py --from_non_trained

You can change the num_skills variable in skills_generator.py to select the amount of skills to generate,
and traj_len to control the length of 1 trajectory to parse by LLM to generate 1 Skill.

This will create you 3 dirs:

1. captioned_trajectories: which holds the agent's trajectories
   in text and also a video of the trajectory with annotated actions done in each step on the frame.
   context_for_skill_generation
2. context_for_skill_generation: holds the context files given to the LLM to generate each skill.
3. option_policies_descriptions: Contains txt files with the option policies(skills) descriptions.

## Training:

For training, move to the project root directory.
Update in main.ipynb:

- Define use_skills boolean as True if want to use skills and use_skills=False if not.
- If use_skills== True: update skills_descriptions_dir to point on the directory with the texts of the option policies you want to use.
  If want to use the skills generated from the Generation section use:
  skills_descriptions_dir = "./skills_gen/option_policies_descriptions"
- You can use the default values or define in main.ipynb the: number of steps to train(steps var), every how many steps to save checkpoint(save_freq)
  or evaluate the model(eval_freq), how many episodes in each evaluation(n_eval_episodes)
- Run all main.ipynb cells
- In the tensorboard cell you can click on the link to see graphs of training and evaluation of the model.
- A results dir will be created, you can see there the training and evaluation figures(mean reward evaluations, episode len and more), checkpoints, logs and more
