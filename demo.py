import ray.rllib.agents.a3c as a3c
import ray.rllib.agents.ppo as ppo
import ray
import models
import env
import env2
import time
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog


ModelCatalog.register_custom_model("my_model", models.TFCNN2)

def env_creator(env_config):
    return env2.DummyPomFFA(env_config)

register_env("my_env",env_creator)

ray.init()

if __name__ == '__main__':

    env_config = {
        "num_rigid": 0,
        # "num_wood":16,
        # "num_items":4
        "reward":{
            "version":"v3"
        }
    }
    env = env2.PomFFA(env_config)

    model_path = "/Users/jiarongqiu/ray_results/A2C_my_env_2020-04-12_21-36-42vs43fldq/checkpoint_7005/checkpoint-7005"

    config = a3c.DEFAULT_CONFIG.copy()
    # config = ppo.DEFAULT_CONFIG.copy()
    config["num_gpus"] = 0
    config["num_workers"] = 0
    config["eager"] = False
    config["env_config"] = {
        "is_training":False
    }
    config["model"] = {
        "custom_model": "my_model",
        "custom_options": {},  # extra options to pass to your model
    }

    trainer = a3c.A2CTrainer(env="my_env", config=config)
    # trainer = ppo.PPOTrainer(env="my_env", config=config)
    trainer.restore(model_path)

    time.sleep(1)
    for i in range(5):
        obs = env.reset()
        env.render()
        done = False
        total_reward = 0
        # for i in range(20):
        while not done:
            actions = trainer.compute_action(obs,full_fetch=True,)
            action = actions[0]

            obs, reward, done, info = env.step(action)
            total_reward += reward
            strength = info['blast_strength']
            # print("action {} strength {} reward {} total {}".format(actions, strength, reward, total_reward))

            if action ==5 :
                print("action {} strength {} reward {} total {}".format(actions,strength,reward,total_reward))
                print(info['board'])
            env.render()
        time.sleep(1)

