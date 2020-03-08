import ray.rllib.agents.a3c as a3c
import ray
import models
import env
import time
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog


ModelCatalog.register_custom_model("my_model", models.TorchVGG)

def env_creator(env_config):
    return env.PomFFA(env_config)

register_env("my_env",env_creator)

ray.init()

if __name__ == '__main__':

    env_config = {
        "num_rigid": 16,
        "num_wood":16,
        "num_items":4
    }
    env = env.PomFFA(env_config)
    obs = env.reset()

    model_path = "/Users/jiarongqiu/Desktop/CS599/results/ray_results/A2C_my_env_2020-03-05_05-41-3158eawp92/checkpoint_601/checkpoint-601"

    config = a3c.DEFAULT_CONFIG.copy()
    config["num_gpus"] = 0
    config["num_workers"] = 1
    config["eager"] = False
    config["env_config"] = {
        "is_training":False
    }
    config["model"] = {
        "custom_model": "my_model",
        "custom_options": {},  # extra options to pass to your model
    }

    trainer = a3c.A2CTrainer(env="my_env", config=config)
    trainer.restore(model_path)
    done = False
    for i in range(100):
        env.render()
        actions = trainer.compute_action(obs)
        print("Action:",actions)
        obs, reward, done, _  = env.step(actions)
        if done:break
        time.sleep(0.5)
    env.render()
    time.sleep(5)
