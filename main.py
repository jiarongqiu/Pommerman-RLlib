import env
import ray
import ray.rllib.agents.ppo as ppo
import models
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog

ModelCatalog.register_custom_model("my_model", models.FullyConnectedNetwork)

def env_creator(env_config):
    return env.PomFFA(env_config)

register_env("my_env",env_creator)

ray.init()


config = ppo.DEFAULT_CONFIG.copy()
config["num_gpus"] = 0
config["num_workers"] = 1
config["eager"] = False
config["use_pytorch"] = True
config["model"]={
        "custom_model": "my_model",
        "custom_options": {},  # extra options to pass to your model
}
print(config)
trainer = ppo.PPOTrainer(env="my_env",config=config)

# Can optionally call trainer.restore(path) to load a checkpoint.

for i in range(1000):
   result = trainer.train()
   print(pretty_print(result))

   if i % 50 == 0:
       checkpoint = trainer.save()
       print("checkpoint saved at", checkpoint)
checkpoint = trainer.save()
print("checkpoint saved at", checkpoint)