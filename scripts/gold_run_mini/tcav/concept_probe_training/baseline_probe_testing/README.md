### Baseline probe training and testing, GoldRun

Trains baseline concept probes on data from different approaches, different batches, and different sample sizes.

Probes are assessed both with a traditional train/test-split (80%/20%), and with a curated validation set. The validation set balances at minimum the number of positive vs negative examples, but preferably also data collection approach sources and batch sources.

In the case of GoldRun, the validation sets per concept contain 10 000 examples. For concept *gold_above*, the validation set is balanced over positive vs negative examples, data collection approaches, and batches. For concept *lava_1_above*, the validation set is only balanced over positive vs negative examples. The latter also lacks examples from approach *model_of_interest_greedy_play*, as this approach amountet to no examples of the concept.