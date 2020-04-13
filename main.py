import env2
import ray
import ray.rllib.agents.a3c as a3c
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.dqn as dqn
import models
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog

ModelCatalog.register_custom_model("my_model", models.TFCNN2)

def env_creator(env_config):
    return env2.PomFFA(env_config)


register_env("my_env", env_creator)

ray.init(memory=10000*1024*1024,object_store_memory=8000*1024*1024,redis_max_memory=8000*1024*1024,driver_object_store_memory=4000*1024*1024)
# ray.init()

config = a3c.DEFAULT_CONFIG.copy()
# config = ppo.DEFAULT_CONFIG.copy()
config["num_gpus"] = 1
config["num_workers"] = 4
config['lr_schedule'] = [[0, 1e-3], [1e7, 1e-5]]
config["train_batch_size"]=1024
config["sample_batch_size"]=512
# config["entropy_coeff"]=0

# config["train_batch_size"]=4096
# config["sgd_minibatch_size"]=256
# config["sample_batch_size"]=256
config["gamma"]=0.995
# config["entropy_coeff"] = 0.01


config['env_config'] = {
    # "num_rigid": 0,
    "reward":{"version":"v3"},
}
config["model"] = {
    "custom_model": "my_model",
    "custom_options": {},  # extra options to pass to your model
}

trainer = a3c.A2CTrainer(env="my_env", config=config)

# trainer = ppo.PPOTrainer(env="my_env", config=config)
# trainer = dqn.DQNTrainer(env="my_env", config=config)
policy = trainer.get_policy()
print(policy.model.base_model.summary())
model_path = "/home/charlieqiu818_gmail_com/ray_results/A2C_my_env_2020-04-12_21-36-42vs43fldq/checkpoint_7005/checkpoint-7005"
trainer.restore(model_path)

for i in range(10000):
    result = trainer.train()
    print(pretty_print(result))

    if i % 50 == 0:
        checkpoint = trainer.save()
        print("checkpoint saved at", checkpoint)

