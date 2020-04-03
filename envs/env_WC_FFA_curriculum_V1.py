import gym
import pommerman
import math
import numpy as np
from pommerman.envs.v0 import Pomme
from pommerman import agents, constants
from gym import spaces
from agents.UtilAgents import HoldAgent


class PomFFA(gym.Env):
    agent_list = [HoldAgent(), HoldAgent(), HoldAgent(), HoldAgent()]
    all_obs = None
    all_action = None
    cur_obs = None
    alive_agents = [10, 11, 12, 13]
    player_agent_id = 10

    def __init__(self, env_config=None):

        pomme_config = pommerman.configs.ffa_competition_env()

        if env_config:
            for k, v in env_config.items():
                if k in pomme_config['env_kwargs']:
                    pomme_config['env_kwargs'][k] = v

        print("pomme_config: ")
        print(pomme_config['env_kwargs'])

        self.pomme = Pomme(**pomme_config['env_kwargs'])

        self.observation_space = self.init_observation_space(pomme_config['env_kwargs'])
        self.action_space = self.pomme.action_space

        self.total_reward = 0
        self.prev_alive = 4
        self.visited = np.zeros(shape=(11, 11))

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
        self.all_obs = obs
        obs = self.get_for_training_agent(obs)
        self.cur_obs = obs
        obs = self.preproess(obs)
        self.total_reward = 0
        self.prev_alive = 4
        self.visited = np.zeros(shape=(11, 11))
        return obs

    def get_reward(self, obs, action, agent_id):
        if len(obs["alive"]) == 1:
            # An agent won. Give them +1, others -1.
            if agent_id in obs['alive']:
                return 1.0 - self.total_reward
            else:
                return -0.5

        if obs["step_count"] >= 500:
            # Game is over from time. Everyone gets -1.
            return -0.5

        # Game running: 0 for alive, -1 for dead.
        if agent_id not in obs['alive']:
            return -0.5

        x, y = obs["position"]
        blast = obs["bomb_blast_strength"]

        px = [1, -1, 0, 0]
        py = [0, 0, -1, 1]

        sum_reward = 0.0

        sum_reward += 20 * (len(obs["alive"]) - self.prev_alive)
        self.prev_alive = len(obs["alive"])

        if action == 0:
            sum_reward -= 0.1

        elif action == 5:
            # sum_reward += 1
            for i in range(4):
                tx = x + px[i]
                ty = y + py[i]
                if tx < 0 or tx > 10 or ty < 0 or ty > 10:
                    continue
                if obs["board"][tx][ty] == 1:
                    sum_reward += 2
                elif obs["board"][tx][ty] > 10:
                    sum_reward += 4
        else:
            assert(1 <= action <= 4), str(action)
            dx = x + px[action-1]
            dy = y + py[action-1]
            if (not (dx < 0 or dx > 10 or dy < 0 or dy > 10)) and obs["board"][dx][dy] == 0:
                if self.visited[dx][dy] > 0:
                    sum_reward -= 0.1
                else:
                    sum_reward += 0.3
                    self.visited[dx][dy] = 1


        sum_reward = sum_reward * 1.0 / 100.0
        new_total_reward = self.total_reward + sum_reward
        if new_total_reward > 0.8 or new_total_reward < -0.5:
            sum_reward = 0.0
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

        self.all_obs = obs
        obs = self.get_for_training_agent(obs)
        self.cur_obs = obs
        reward = self.get_reward(self.cur_obs, action, self.player_agent_id)
        self.alive_agents = obs['alive']
        if (self.player_agent_id not in self.alive_agents) or obs["step_count"] >= 500:
            done = True
        obs = self.preproess(obs)
        return obs, reward, done, {}

    def get_for_training_agent(self, inputs):
        order = self.player_agent_id - 10
        return inputs[order].copy()

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
        board_size = env_config['board_size'] or 11
        num_items = env_config['num_items'] or 11
        print("env config: {}".format(env_config))
        # board_size = 11

        board = spaces.Box(low=0, high=len(constants.Item), shape=(board_size, board_size))
        danger = spaces.Box(low=0, high=20, shape=(board_size, board_size))
        bomb_blast_strength = spaces.Box(low=0, high=num_items, shape=(board_size, board_size))
        bomb_life = spaces.Box(low=0, high=9, shape=(board_size, board_size))
        flame_life = spaces.Box(low=0, high=10, shape=(board_size, board_size))
        position = spaces.Box(low=0, high=board_size, shape=(2,))
        blast_strength = spaces.Box(low=1, high=num_items, shape=(1,))
        ammo = spaces.Box(low=0, high=num_items, shape=(1,))
        # return spaces.Dict({"board": board,
        #                     "bomb_blast_strength": bomb_blast_strength, "bomb_life": bomb_life,
        #                     "flame_life": flame_life,
        #                     "position": position, "ammo": ammo, "blast_strength": blast_strength})
        return spaces.Dict({"board": board,
                            "bomb_blast_strength": bomb_blast_strength, "bomb_life": bomb_life,
                            "flame_life": flame_life,
                            "position": position, "ammo": ammo, "blast_strength": blast_strength,
                            "danger": danger})

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

        board = obs['board']
        bomb_blast_strength = obs['bomb_blast_strength']
        bomb_life = obs['bomb_life']
        # flame_life = obs['flame_life']
        # position = obs['position']
        # ammo = obs['ammo']
        # blast_strength = obs['blast_strength']

        danger = np.ndarray(shape=(11, 11), dtype=int)

        for x in range(11):
            for y in range(11):
                danger[x][y] = 10
                if board[x][y] == 4:
                    board[x][y] = 0
                    danger[x][y] = 0
                elif board[x][y] == 3:
                    board[x][y] = 0
                elif board[x][y] == 10:
                    board[x][y] = 1
                elif board[x][y] > 10:
                    board[x][y] = 5
                elif 6 <= board[x][y] <= 8:
                    board[x][y] = 3
                elif board[x][y] == 1:
                    board[x][y] = 4

        for x in range(11):
            for y in range(11):
                if bomb_life[x][y] > 0:
                    strength = int(bomb_blast_strength[x][y]+0.5)
                    for tx in range(max(0, x-strength+1), min(11, x+strength)):
                        danger[tx][y] = min(danger[tx][y], bomb_life[x][y])
                    for ty in range(max(0, y-strength+1), min(11, y+strength)):
                        danger[x][ty] = min(danger[x][ty], bomb_life[x][y])

        obs['danger'] = danger

        return obs

    def render(self):
        self.pomme.render()
