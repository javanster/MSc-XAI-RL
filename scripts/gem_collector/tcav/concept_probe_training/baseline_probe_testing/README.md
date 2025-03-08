### Baseline probe training and testing, GemCollector

Trains baseline concept probes on data from different approaches, different batches, and different sample sizes.

Probes are assessed both with a traditional train/test-split (80%/20%), and with a curated validation set. The validation set balances at minimum the number of positive vs negative examples, but preferably also data collection approach sources and batch sources.

In the case of GemCollector, the validation set per concept contains 10 000 examples, balanced over positive vs negative examples, data collection approaches, and batches.