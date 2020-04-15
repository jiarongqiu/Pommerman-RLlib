import gym
import pommerman
import numpy as np
from pommerman.envs.v0 import Pomme
from pommerman import agents, constants
from gym import spaces
from agents.UtilAgents import HoldAgent


class PomFFA(gym.Env):

    def __init__(self, env_config=None):

        self.agent_list = [HoldAgent(), agents.SimpleAgent(), HoldAgent(), HoldAgent()]
        # self.agent_list = [agents.SimpleAgent(), agents.SimpleAgent(), agents.SimpleAgent(), agents.RandomAgent()]
        self.all_obs = None
        self.all_action = None
        self.cur_obs = None
        self.alive_agents = [10, 11, 12, 13]
        self.player_agent_id = 10
        self.total_reward = 0

        pomme_config = pommerman.configs.ffa_competition_env()

        if env_config:
            for k, v in env_config.items():
                if k in pomme_config['env_kwargs']:
                    pomme_config['env_kwargs'][k] = v

        self.pomme = Pomme(**pomme_config['env_kwargs'])

        self.observation_space = self.init_observation_space(pomme_config['env_kwargs'])
        self.action_space = self.pomme.action_space

        if not env_config or (env_config and env_config.get("is_training", True)):
            # initialize env twice could raise error here.
            self.init(pomme_config)

    def init(self, pomm_config):
        for id_, agent in enumerate(self.agent_list):
            assert isinstance(agent, agents.BaseAgent)
            print(id_, pomm_config['game_type'])
            agent.init_agent(id_, pomm_config['game_type'])
        self.pomme.set_agents(self.agent_list)
        self.pomme.set_init_game_state(None)

    def reset(self):
        obs = self.pomme.reset()
        self.all_obs = obs.copy()
        obs = self.get_for_training_agent(obs)
        self.cur_obs = obs.copy()
        obs = self.preproess(obs)
        self.total_reward = 0
        return obs

    def get_reward(self, obs, action, agent_id):
        if len(obs["alive"]) == 1:
            # An agent won. Give them +1, others -1.
            if agent_id in obs['alive']:
                return 0.5
            else:
                return -0.5

        if obs["step_count"] >= 500:
            # Game is over from time. Everyone gets -1.
            return -0.5

        # Game running: 0 for alive, -1 for dead.
        if agent_id not in obs['alive']:
            return -0.5

        x, y = obs["position"]
        # blast = obs["bomb_blast_strength"]

        px = [0, 1, 0, -1]
        py = [1, 0, -1, 0]

        sum_reward = 0
        if action == 5:
            for i in range(4):
                tx = x+px[i]
                ty = y+py[i]
                if tx<0 or tx>10 or ty<0 or ty>10:
                    continue
                if obs["board"][tx][ty] == 1:
                    sum_reward += 1
                elif obs["board"][tx][ty] > 10:
                    sum_reward += 4

        sum_reward = sum_reward*1.0/200.0
        new_total_reward = self.total_reward + sum_reward
        if new_total_reward > 0.5 or new_total_reward < -0.5:
            sum_reward = 0
        else:
            self.total_reward = new_total_reward

        return sum_reward

    def step(self, action):
        actions = self.pomme.act(self.all_obs)
        if self.alive_agents and self.player_agent_id in self.alive_agents:
            actions = self.set_for_training_agent(actions, action)
        else:
            actions = self.set_for_training_agent(actions, 0)
        obs, rewards, done, info = self.pomme.step(actions)

        # print(obs)
        del self.all_obs
        self.all_obs = obs.copy()
        obs = self.get_for_training_agent(obs)
        del self.cur_obs
        self.cur_obs = obs.copy()
        reward = self.get_reward(self.cur_obs, action, self.player_agent_id)
        self.alive_agents = obs['alive']

        if self.player_agent_id not in self.alive_agents or self.cur_obs["step_count"] >= 500:
            done = True
        obs = self.preproess(obs)

        return obs, reward, done, {}

    def get_for_training_agent(self, inputs):
        order = self.player_agent_id - 10
        return inputs[order]

    def set_for_training_agent(self, inputs, value):
        order = self.player_agent_id - 10
        inputs[order] = value
        return inputs

    def init_observation_space(self, env_config):
        """
            observations for agents
            board: n^2
            bomb blast strength: n^2
            bomb life: n^2
        """
        board_size = env_config['board_size']
        num_items = env_config['num_items']
        # print("env config: {}".format(env_config))
        # board_size = 11

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
