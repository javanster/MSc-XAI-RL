import wandb

from .ccr_ppo_param_sweep_init import train

# DEFINE BEFORE RUNNING
SWEEP_ID = ""  # Get ID by initating a sweep from script 'gc_ddqn_param_sweep_init'
ENTITY = "jsterkenburg"  # wandb entity
PROJECT = "CCR"


if __name__ == "__main__":
    if any(len(attr) == 0 for attr in [SWEEP_ID, ENTITY, PROJECT]):
        raise ValueError(
            "SWEEP_ID, ENTITY and PROJECT must be defined appropriately before running a sweep"
        )

    wandb.login()  # type: ignore
    wandb.agent(sweep_id=SWEEP_ID, function=train, entity=ENTITY, project=PROJECT)  # type: ignore
