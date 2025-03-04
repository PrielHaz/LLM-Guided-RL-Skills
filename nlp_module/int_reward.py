# To be implemented according the article.
# Give intrinsic rewards to the meta policy for selecting
# the option policy that adequate for the current state semantically.

# option 1: calc similarity between transition made and the option policy description.
# option 2: calc generate goals proposals, then for each skill calc the max similarity to some goal - that's the skill score.
# Normalize the scores to mean=0 and std=1, multiply them by lambda_semantic, and r_intrinsic will be the normalized
# score of the chosen skill, punishing on selecting skills that are less relevant from the mean and rewarding for selecting
# skills that are more relevant from the mean for current state.
