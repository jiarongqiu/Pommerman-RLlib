import gym
import pommerman
import numpy as np
import agents
import copy
import random
from pommerman.envs.v0 import Pomme
from pommerman import constants
from gym import spaces
from reward import Reward
from utils import *

class PomFFA(gym.Env):
    agent_list = [agents.StaticAgent(), agents.StaticAgent(), agents.StaticAgent(), agents.StaticAgent()]
    alive_agents = [10, 11, 12, 13]
    agent_id = 10
    ammo = 1
    blast_strength = 2
    state = {}

    def __init__(self, env_config={}):
        pomme_config = pommerman.configs.ffa_competition_env()
        self.reward = Reward(env_config.get("reward"))

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
        self.init_state()

    def init_state(self):
        self.state['agent_id'] = self.agent_id
        self.state['alive'] = self.alive_agents
        self.state['visited'] = set()
        self.state['blast_strength'] = self.blast_strength
        self.state['ammo'] = self.ammo
        self.state["bombs"] = {}

    def reset(self):
        all_obs = self.pomme.reset()
        obs = self.get_for_training_agent(all_obs)
        self.init_state()

        self.state['prev_obs'] = copy.deepcopy(obs)
        self.state['all_obs'] = all_obs
        self.state['alive'] = obs['alive']

        obs = self.build_obs(obs,self.state)
        return obs

    def step(self, action):
        actions = self.pomme.act(self.state['all_obs'])
        actions = self.set_for_training_agent(actions, action)

        all_obs, _, _, _ = self.pomme.step(actions)
        obs = self.get_for_training_agent(all_obs)
        info = {'board':obs['board'],'blast_strength':obs['blast_strength']}
        done = self.get_done(obs)
        reward, self.state = self.reward.get_reward(action, obs, self.state)

        self.state['prev_obs'] = copy.deepcopy(obs)
        self.state['all_obs'] = all_obs
        self.state['alive'] = obs['alive']
        self.state['blast_strength'] = obs['blast_strength']
        self.state['ammo'] = obs['ammo']

        obs = self.build_obs(obs,self.state)
        return obs, reward, done, info

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

    def build_obs(self,obs,state):
        board = obs['board']
        bomb_blast_strength = obs['bomb_blast_strength']
        bomb_life = obs['bomb_life']
        flame_life = obs['flame_life']
        agent_id = state['agent_id']
        ammo = state['ammo']
        passage = np.zeros_like(board)
        wall = np.zeros_like(board)
        wood = np.zeros_like(board)
        bomb = np.zeros_like(board)
        bonus = np.zeros_like(board)
        me = np.zeros_like(board)
        enemy = np.zeros_like(board)
        for y in range(board.shape[0]):
            for x in range(board.shape[1]):
                v = board[y][x]
                if v == 0:
                    passage[y][x] = 1
                elif v == 1:
                    wall[y][x] = 1
                elif v == 2:
                    wood[y][x] = 1
                elif v == 3:
                    bomb = create_cross(bomb, (y, x), bomb_blast_strength[y][x])
                elif v == 4:
                    pass
                elif v == 6 or v == 7:
                    bonus[y][x] = 1
                elif v >= 10:
                    if v == agent_id:
                        me[y][x] = 1
                    else:
                        enemy[y][x] = 1
                    if bomb_blast_strength[y][x] > 0:
                        bomb = create_cross(bomb, (y, x), bomb_blast_strength[y][x])

        ammo = ammo * np.ones_like(board) / 12
        bomb_life /= 9
        flame_life /= 3
        board = np.transpose(np.stack([passage,wall,wood,bomb,bonus,me,enemy,bomb_life,flame_life,ammo]),[1,2,0])
        return board



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

        board = spaces.Box(low=0, high=1, shape=(board_size, board_size,10)) # passage,wall,wood,bomb,bonus,me,enemies
        bomb_life = spaces.Box(low=0, high=9, shape=(board_size, board_size))
        flame_life = spaces.Box(low=0, high=3, shape=(board_size, board_size))
        ammo = spaces.Box(low=0, high=num_items, shape=(board_size,board_size))
        # return spaces.Dict({"board": board, "bomb_life": bomb_life, "flame_life": flame_life,"ammo": ammo})
        return board

    @staticmethod
    def init_action_space():
        return spaces.Discrete(6)

    def render(self):
        self.pomme.render()


class DummyPomFFA(gym.Env):

    def __init__(self, env_config):
        pomme_config = pommerman.configs.ffa_competition_env()
        self.observation_space = PomFFA.init_observation_space(pomme_config['env_kwargs'])
        self.action_space = PomFFA.init_action_space()
        self.fake_obs =  np.zeros((11, 11,10))

    def reset(self):
        return self.fake_obs

    def step(self, action):
        return self.fake_obs, 0, False, {}


if __name__ == '__main__':
    env = PomFFA({"reward": {"version": "v1"}})

    obs = env.reset()
    for i in range(10):
        action = random.randint(1, 5)
        obs, reward, done, info = env.step(action)
        print(info['board'],action,reward)


    #
    #
    # total = -1
    # path = []
    # while total < -0.99:
    #     total = 0
    #     done = False
    #     step = 0
    #     path = []
    #     obs = env.reset()
    #
    #     while not done:
    #         action = random.randint(1, 5)
    #         obs, reward, done, _ = env.step(action)
    #         path.append([step,action,reward])
    #         step += 1
    #         total += reward
    #     print(done,total)
    # print(total,path)
