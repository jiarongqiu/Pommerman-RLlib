import gym
import pommerman
import numpy as np
import time
import agents
import copy
import random
from pommerman.envs.v0 import Pomme
from pommerman import constants
from gym import spaces
from reward import Reward


class PomFFA(gym.Env):
    agent_list = [agents.StaticAgent(), agents.StaticAgent(), agents.StaticAgent(), agents.StaticAgent()]
    all_obs = None
    all_action = None
    pre_obs = None
    alive_agents = [10, 11, 12, 13]
    agent_id = 10
    state = {
    }

    def __init__(self, env_config=None):

        pomme_config = pommerman.configs.ffa_competition_env()

        if env_config:
            for k, v in env_config.items():
                if k in pomme_config['env_kwargs']:
                    pomme_config['env_kwargs'][k] = v
            self.reward = Reward(env_config.get("reward"))
        else:
            self.reward = Reward()

        print("Pommerman Config:", pomme_config['env_kwargs'])

        self.pomme = Pomme(**pomme_config['env_kwargs'])

        self.observation_space = self.init_observation_space(pomme_config['env_kwargs'])
        self.action_space = self.pomme.action_space

        if not env_config or (env_config and env_config.get("is_training", True)):
            # initialize env twice could raise error here.
            self.init(pomme_config)

    def init(self, pomm_config):
        for id_, agent in enumerate(self.agent_list):
            assert isinstance(agent, agents.BaseAgent)
            agent.init_agent(id_, pomm_config['game_type'])
        self.pomme.set_agents(self.agent_list)
        self.pomme.set_init_game_state(None)

    def reset(self):
        obs = self.pomme.reset()
        self.all_obs = obs
        obs = self.get_for_training_agent(obs)
        state = {
            "prev_obs": None,
            "visited": set(),
            "agent_id": 10,
            "alive":[10, 11, 12, 13],
            "strength":2,
            "ammo":1,
            "bombs":{},
        }
        state['prev_obs'] = copy.deepcopy(obs)
        state['position'] = obs['position']
        self.state = state
        obs = self.preproess(obs)
        return obs

    def step(self, action):
        actions = self.pomme.act(self.all_obs)
        actions = self.set_for_training_agent(actions, action)

        obs, rewards, _, _ = self.pomme.step(actions)
        self.all_obs = obs
        obs = self.get_for_training_agent(obs)
        reward, self.state = self.reward.get_reward(action, obs, self.state)
        done = self.get_done(obs)
        self.state['prev_obs'] = copy.deepcopy(obs)
        self.state['position'] = obs['position']
        obs = self.preproess(obs)

        return obs, reward, done, {}

    def get_for_training_agent(self, inputs):
        order = self.agent_id - 10
        return inputs[order]

    def set_for_training_agent(self, inputs, value):
        order = self.agent_id - 10
        inputs[order] = value
        return inputs

    def get_done(self,obs):
        if self.agent_id not in obs['alive']:
            return True
        if obs['step_count'] >= 800:
            return True
        return False

    @staticmethod
    def init_observation_space(env_config):
        """
            observations for agents
            board: n^2
            bomb blast strength: n^2
            bomb life: n^2
        """
        board_size = env_config['board_size']
        num_items = env_config['num_items']

        board = spaces.Box(low=0, high=len(constants.Item), shape=(board_size, board_size))
        bomb_blast_strength = spaces.Box(low=0, high=num_items, shape=(board_size, board_size))
        bomb_life = spaces.Box(low=0, high=9, shape=(board_size, board_size))
        flame_life = spaces.Box(low=0, high=3, shape=(board_size, board_size))
        position = spaces.Box(low=0, high=board_size, shape=(2,))
        blast_strength = spaces.Box(low=1, high=num_items, shape=(1,))
        ammo = spaces.Box(low=0, high=num_items, shape=(1,))
        return spaces.Dict({"board": board, "bomb_blast_strength": bomb_blast_strength, "bomb_life": bomb_life,
                            "flame_life": flame_life,
                            "position": position, "ammo": ammo, "blast_strength": blast_strength})

    @staticmethod
    def init_action_space():
        return spaces.Discrete(6)

    @staticmethod
    def preproess(obs):
        del obs["game_type"]
        del obs["game_env"]
        del obs["can_kick"]
        del obs["teammate"]
        del obs["enemies"]
        del obs["step_count"]
        del obs['alive']
        del obs['bomb_moving_direction']

        obs['position'] = np.array(obs['position'])
        obs['ammo'] = np.array([obs['ammo']])
        obs['blast_strength'] = np.array([obs['blast_strength']])

        return obs

    def render(self):
        self.pomme.render()


class DummyPomFFA(gym.Env):

    def __init__(self, env_config):
        pomme_config = pommerman.configs.ffa_competition_env()
        self.observation_space = PomFFA.init_observation_space(pomme_config['env_kwargs'])
        self.action_space = PomFFA.init_action_space()
        self.fake_obs = {
            "board": np.zeros((11, 11)),
            "bomb_blast_strength": np.zeros((11, 11)),
            "bomb_life": np.zeros((11, 11)),
            "flame_life": np.zeros((11, 11)),
            "position": np.array([0, 0]),
            "ammo": np.array([0]),
            "blast_strength": np.array([1])
        }

    def reset(self):
        return self.fake_obs

    def step(self, action):
        return self.fake_obs, 0, False, {}


if __name__ == '__main__':
    env = PomFFA({"reward": {"version": "v2"}})


    total = -2
    path = []
    while total < -1.95:
        total = 0
        done = False
        step = 0
        path = []
        obs = env.reset()

        while not done:
            action = random.randint(1, 5)
            obs, reward, done, _ = env.step(action)
            path.append([step,action,reward,obs['position']])
            step += 1
            total += reward
        print(done,total)
    print(total,path)
