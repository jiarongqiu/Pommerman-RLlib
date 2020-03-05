import time
import sys
import ray
import ray.rllib.agents.a3c as a3c

from ray.tune.logger import pretty_print
from ray.tune.registry import register_env

from envs.env_WC_FFA_V1 import PomFFA
# from env import PomFFA

def env_creator(env_config):
    return PomFFA(env_config)


register_env("pom", env_creator)
ray.init()
model_config = {
    # === Built-in options ===
    # Filter config. List of [out_channels, kernel, stride] for each filter
    "conv_filters": None,
    # Nonlinearity for built-in convnet
    "conv_activation": "relu",
    # Nonlinearity for fully connected net (tanh, relu)
    "fcnet_activation": "tanh",
    # Number of hidden layers for fully connected net
    "fcnet_hiddens": [256, 128, 32],
    # For control envs, documented in ray.rllib.models.Model
    "free_log_std": False,
    # Whether to skip the final linear layer used to resize the hidden layer
    # outputs to size `num_outputs`. If True, then the last hidden layer
    # should already match num_outputs.
    "no_final_linear": False,
    # Whether layers should be shared for the value function.
    "vf_share_layers": True,

    # == LSTM ==
    # Whether to wrap the model with a LSTM
    "use_lstm": False,
    # Max seq len for training the LSTM, defaults to 20
    "max_seq_len": 20,
    # Size of the LSTM cell
    "lstm_cell_size": 256,
    # Whether to feed a_{t-1}, r_{t-1} to LSTM
    "lstm_use_prev_action_reward": False,
    # When using modelv1 models with a modelv2 algorithm, you may have to
    # define the state shape here (e.g., [256, 256]).
    "state_shape": None,

    # == Atari ==
    # Whether to enable framestack for Atari envs
    "framestack": True,
    # Final resized frame dimension
    "dim": 84,
    # (deprecated) Converts ATARI frame to 1 Channel Grayscale image
    "grayscale": False,
    # (deprecated) Changes frame to range from [-1, 1] if true
    "zero_mean": True,

    # === Options for custom models ===
    # Name of a custom model to use
    "custom_model": None,
    # Name of a custom action distribution to use.
    "custom_action_dist": None,

    # Extra options to pass to the custom classes
    "custom_options": {},
    # Custom preprocessors are deprecated. Please use a wrapper class around
    # your environment instead to preprocess observations.
    "custom_preprocessor": None,
}


def game_train():
    config = a3c.DEFAULT_CONFIG.copy()
    config["num_gpus"] = 0
    config["num_workers"] = 1
    config["eager"] = False
    config["use_pytorch"] = True
    config["model"] = model_config
    print(config)
    trainer = a3c.A2CTrainer(env="pom", config=config)

    # Can optionally call trainer.restore(path) to load a checkpoint.

    for i in range(10000):
        result = trainer.train()
        print(pretty_print(result))

        if i % 200 == 0:
            checkpoint = trainer.save()
            print("checkpoint saved at", checkpoint)


def game_eval():
    env = PomFFA()
    obs = env.reset()

    model_path = "/Users/jiarongqiu/ray_results/A2C_my_env_2020-03-03_21-14-17xifin5lb/checkpoint_1/checkpoint-1"

    config = a3c.DEFAULT_CONFIG.copy()
    config["num_gpus"] = 1
    config["num_workers"] = 1
    config["eager"] = False
    config["use_pytorch"] = True
    config["env_config"] = {
        "is_training": False
    }
    config["model"] = model_config

    trainer = a3c.A2CTrainer(env="pom", config=config)
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


entrypoint = next(iter(sys.argv[1:]), "game_eval")
locals()[entrypoint]()
