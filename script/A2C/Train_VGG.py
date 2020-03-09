import time
import sys
import ray
import ray.rllib.agents.a3c as a3c
from ray.rllib.agents.ppo import ppo
from ray.rllib.models import ModelCatalog
from ray.rllib.models.catalog import MODEL_DEFAULTS

from ray.tune.logger import pretty_print
from ray.tune.registry import register_env

from envs.env_WC_FFA_V1_single_enemy import PomFFA
from models.tf_cnn import TFCNN


def env_creator(env_config):
    return PomFFA(env_config)


def game_train():
    config = a3c.DEFAULT_CONFIG.copy()
    config["num_gpus"] = 1
    config["num_workers"] = 12
    # config["eager"] = False
    # config["use_pytorch"] = False
    config["model"] = model_config
    config['env_config'] = {
        "reward": {"version": "v0"},
        "num_rigid": 16,
        "num_wood": 16,
        "num_items": 4
    }
    trainer = a3c.A3CTrainer(env="pom", config=config)
    # trainer = ppo.PPOTrainer(env="pom", config=config)

    # Can optionally call trainer.restore(path) to load a checkpoint.

    for i in range(40000):
        result = trainer.train()
        print(pretty_print(result))
        del result

        if i % 100 == 0:
            checkpoint = trainer.save()
            print("checkpoint saved at", checkpoint)
            del checkpoint


def game_eval():
    env = PomFFA()
    obs = env.reset()

    model_path = "/home/subill/ray_results/A2C_pom_2020-03-05_01-05-33zg_fnulc/checkpoint_501/checkpoint-501"

    config = a3c.DEFAULT_CONFIG.copy()
    config["num_gpus"] = 1
    config["num_workers"] = 1
    # config["eager"] = False
    # config["use_pytorch"] = True
    config["env_config"] = {
        "is_training": False
    }
    config["model"] = model_config

    trainer = a3c.A3CTrainer(env="pom", config=config)
    trainer.restore(model_path)
    for i in range(500):
        env.render()
        actions = trainer.compute_action(obs)
        print(actions)
        obs, reward, done, _ = env.step(actions)
        if done:
            break
        time.sleep(0.5)

    env.render()
    time.sleep(10)


if __name__ == "__main__":
    # ray.init(memory=11*1024*1024*1024, object_store_memory=5*1024*1024*1024)
    ray.init()
    ModelCatalog.register_custom_model("my_model", TFCNN)
    register_env("pom", env_creator)
    model_config = MODEL_DEFAULTS.copy()
    model_config["custom_model"] = "my_model"
    model_config["custom_options"] = ""
    entrypoint = next(iter(sys.argv[1:]), "game_eval")
    locals()[entrypoint]()
