import env
import ray
import ray.rllib.agents.a3c as a3c
import ray.rllib.agents.ppo as ppo
import models
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog

ModelCatalog.register_custom_model("my_model", models.VGG)


def env_creator(env_config):
    return env.PomFFA(env_config)


register_env("my_env", env_creator)

ray.init()

config = a3c.DEFAULT_CONFIG.copy()
config["num_gpus"] = 0
config["num_workers"] = 1
config["eager"] = False
config["use_pytorch"] = True
config["model"] = {
    "custom_model": "my_model",
    "custom_options": {"num_hiddens": [128, 32]},  # extra options to pass to your model
}
# trainer = a3c.A2CTrainer(env="my_env", config=config)
trainer = ppo.PPOTrainer(env="my_env", config=config)

# Can optionally call trainer.restore(path) to load a checkpoint.

for i in range(6):
    result = trainer.train()
    print(pretty_print(result))

    if i % 5 == 0:
        checkpoint = trainer.save()
        print("checkpoint saved at", checkpoint)

