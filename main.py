import env
import ray
import ray.rllib.agents.a3c as a3c
import ray.rllib.agents.ppo as ppo
import models
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog

ModelCatalog.register_custom_model("my_model", models.TFCNN)


def env_creator(env_config):
    return env.PomFFA(env_config)


register_env("my_env", env_creator)

ray.init()

config = a3c.DEFAULT_CONFIG.copy()
config["num_gpus"] = 0
config["num_workers"] = 1
# config["eager"] = True
# config["use_pytorch"] = False
config['env_config'] = {
    "reward":{"version":"v0"},
    "num_rigid": 16,
    "num_wood":16,
    "num_items":4
}
config["model"] = {
    "custom_model": "my_model",
    "custom_options": {},  # extra options to pass to your model
}
trainer = a3c.A3CTrainer(env="my_env", config=config)
# trainer = ppo.PPOTrainer(env="my_env", config=config)

# Can optionally call trainer.restore(path) to load a checkpoint.

for i in range(1000):
    result = trainer.train()
    print(pretty_print(result))

    if i % 100 == 0:
        checkpoint = trainer.save()
        print("checkpoint saved at", checkpoint)

