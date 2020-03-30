import time
import sys
import ray
import ray.rllib.agents.a3c as a3c
import ray.rllib.agents.ppo as ppo
from ray.rllib.models import ModelCatalog
from ray.rllib.models.catalog import MODEL_DEFAULTS

from ray.tune.logger import pretty_print
from ray.tune.registry import register_env

from envs.env_WC_FFA_V1 import PomFFA
from models.tf_cnn import TFCNN


def env_creator(env_config):
    return PomFFA(env_config)


# DEFAULT_CONFIG = with_common_config({
#     # Should use a critic as a baseline (otherwise don't use value baseline;
#     # required for using GAE).
#     "use_critic": True,
#     # If true, use the Generalized Advantage Estimator (GAE)
#     # with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
#     "use_gae": True,
#     # Size of rollout batch
#     "sample_batch_size": 10,
#     # GAE(gamma) parameter
#     "lambda": 1.0,
#     # Max global norm for each gradient calculated by worker
#     "grad_clip": 40.0,
#     # Learning rate
#     "lr": 0.0001,
#     # Learning rate schedule
#     "lr_schedule": None,
#     # Value Function Loss coefficient
#     "vf_loss_coeff": 0.5,
#     # Entropy coefficient
#     "entropy_coeff": 0.01,
#     # Min time per iteration
#     "min_iter_time_s": 5,
#     # Workers sample async. Note that this increases the effective
#     # sample_batch_size by up to 5x due to async buffering of batches.
#     "sample_async": True,
# })

# default for ppo
# DEFAULT_CONFIG = with_common_config({
#     # Should use a critic as a baseline (otherwise don't use value baseline;
#     # required for using GAE).
#     "use_critic": True,
#     # If true, use the Generalized Advantage Estimator (GAE)
#     # with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
#     "use_gae": True,
#     # The GAE(lambda) parameter.
#     "lambda": 1.0,
#     # Initial coefficient for KL divergence.
#     "kl_coeff": 0.2,
#     # Size of batches collected from each worker.
#     "sample_batch_size": 200,
#     # Number of timesteps collected for each SGD round. This defines the size
#     # of each SGD epoch.
#     "train_batch_size": 4000,
#     # Total SGD batch size across all devices for SGD. This defines the
#     # minibatch size within each epoch.
#     "sgd_minibatch_size": 128,
#     # Whether to shuffle sequences in the batch when training (recommended).
#     "shuffle_sequences": True,
#     # Number of SGD iterations in each outer loop (i.e., number of epochs to
#     # execute per train batch).
#     "num_sgd_iter": 30,
#     # Stepsize of SGD.
#     "lr": 5e-5,
#     # Learning rate schedule.
#     "lr_schedule": None,
#     # Share layers for value function. If you set this to True, it's important
#     # to tune vf_loss_coeff.
#     "vf_share_layers": False,
#     # Coefficient of the value function loss. IMPORTANT: you must tune this if
#     # you set vf_share_layers: True.
#     "vf_loss_coeff": 1.0,
#     # Coefficient of the entropy regularizer.
#     "entropy_coeff": 0.0,
#     # Decay schedule for the entropy regularizer.
#     "entropy_coeff_schedule": None,
#     # PPO clip parameter.
#     "clip_param": 0.3,
#     # Clip param for the value function. Note that this is sensitive to the
#     # scale of the rewards. If your expected V is large, increase this.
#     "vf_clip_param": 10.0,
#     # If specified, clip the global norm of gradients by this amount.
#     "grad_clip": None,
#     # Target value for KL divergence.
#     "kl_target": 0.01,
#     # Whether to rollout "complete_episodes" or "truncate_episodes".
#     "batch_mode": "truncate_episodes",
#     # Which observation filter to apply to the observation.
#     "observation_filter": "NoFilter",
#     # Uses the sync samples optimizer instead of the multi-gpu one. This is
#     # usually slower, but you might want to try it if you run into issues with
#     # the default optimizer.
#     "simple_optimizer": False,
#     # Use PyTorch as framework?
#     "use_pytorch": False
# })

def game_train():
    config = a3c.DEFAULT_CONFIG.copy()
    # config = ppo.ppo.DEFAULT_CONFIG.copy()
    config["num_gpus"] = 1
    config["num_workers"] = 12
    # config["lambda"] = 1.0
    config["model"] = model_config

    # trainer = ppo.PPOTrainer(env="pom", config=config)
    # config["lr"] = 0.0001
    config["lr_schedule"] = [[0, 5e-4], [2000000, 5e-5], [4000000, 1e-5], [6000000, 1e-6], [8000000, 1e-7]]
    # config["vf_clip_param"] = 0.5
    config["grad_clip"] = 0.5

    trainer = a3c.A3CTrainer(env="pom", config=config)
    # trainer = ppo.ppo.PPOTrainer(env="pom", config=config)
    # Can optionally call trainer.restore(path) to load a checkpoint.

    for i in range(30000):
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

    model_path = "/home/subill/ray_results/APPO_pom_2020-03-29_00-05-42_5225xam/checkpoint_3601/checkpoint-3601"

    # config = a3c.DEFAULT_CONFIG.copy()
    config = ppo.appo.DEFAULT_CONFIG.copy()
    config["num_gpus"] = 1
    config["num_workers"] = 1
    config["env_config"] = {
        "is_training": False
    }
    config["model"] = model_config

    # trainer = a3c.A3CTrainer(env="pom", config=config)
    trainer = ppo.appo.APPOTrainer(env="pom", config = config)
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
