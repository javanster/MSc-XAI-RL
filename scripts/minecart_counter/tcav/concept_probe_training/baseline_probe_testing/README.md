### Baseline probe training and testing, MinecartCounter

Trains baseline concept probes on data from different approaches, different batches, and different sample sizes.

Probes are assessed both with a traditional train/test-split (80%/20%), and with a curated validation set, containing no duplicate examples. The validation set balances at minimum the number of positive vs negative examples for binary concepts, but preferably also data collection approach sources and batch sources. For continuous concepts, the validation set is attempted balanced over approaches and batches.