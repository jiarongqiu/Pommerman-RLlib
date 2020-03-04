import gym
import pommerman
import numpy as np
from pommerman.envs.v0 import Pomme
from pommerman import agents,constants
from gym import spaces
from reward import Reward


class PomFFA(gym.Env):

    agent_list = [agents.SimpleAgent(), agents.SimpleAgent(), agents.SimpleAgent(), agents.RandomAgent()]
    all_obs = None
    all_action = None
    cur_obs = None
    alive_agents = [10, 11, 12, 13]
    agent_id = 13

    def __init__(self, env_config=None):

        pomme_config = pommerman.configs.ffa_competition_env()

        if env_config:
            for k, v in env_config.items():
                if k in pomme_config['env_kwargs']:
                    pomme_config['env_kwargs'][k] = v
            self.reward = Reward(env_config.get("reward"))
        else:
            self.reward = Reward()

        print(pomme_config['env_kwargs'])


        self.pomme = Pomme(**pomme_config['env_kwargs'])

        self.observation_space = self.init_observation_space(pomme_config['env_kwargs'])
        self.action_space = self.pomme.action_space

        if not env_config or (env_config and env_config.get("is_training",True)):
            # initialize env twice could raise error here.
            self.init(pomme_config)

    def init(self,pomm_config):
        for id_, agent in enumerate(self.agent_list):
            assert isinstance(agent, agents.BaseAgent)
            print(id_,pomm_config['game_type'])
            agent.init_agent(id_, pomm_config['game_type'])
        self.pomme.set_agents(self.agent_list)
        self.pomme.set_init_game_state(None)

    def reset(self):
        obs = self.pomme.reset()
        self.all_obs = obs
        obs = self.get_for_training_agent(obs)
        self.cur_obs = obs
        obs = self.preproess(obs)
        return obs

    def step(self, action):
        actions = self.pomme.act(self.all_obs)
        if self.alive_agents and 13 in self.alive_agents:
            actions = self.set_for_training_agent(actions, action)
        else:
            actions = self.set_for_training_agent(actions, 0)
        obs, rewards, done, info = self.pomme.step(actions)
        self.all_obs = obs
        obs = self.get_for_training_agent(obs)
        self.cur_obs = obs
        reward = self.reward.get_reward(self.cur_obs, action, self.agent_id)
        self.alive_agents = obs['alive']
        obs = self.preproess(obs)
        return obs, reward, done, {}

    def get_for_training_agent(self, inputs):
        order = self.agent_id - 10
        return inputs[order]

    def set_for_training_agent(self, inputs, value):
        order = self.agent_id - 10
        inputs[order] = value
        return inputs

    def init_observation_space(self,env_config):
        """
            observations for agents
            board: n^2
            bomb blast strength: n^2
            bomb life: n^2
        """
        board_size = env_config['board_size']
        num_items = env_config['num_items']

        board = spaces.Box(low=0,high=len(constants.Item),shape=(board_size,board_size))
        bomb_blast_strength = spaces.Box(low=0,high=num_items,shape=(board_size,board_size))
        bomb_life = spaces.Box(low=0,high=9,shape=(board_size,board_size))
        flame_life = spaces.Box(low=0,high=3,shape=(board_size,board_size))
        position = spaces.Box(low=0,high=board_size,shape=(2,))
        blast_strength = spaces.Box(low=1,high=num_items,shape=(1,))
        ammo = spaces.Box(low=1,high=num_items,shape=(1,))
        return spaces.Dict({"board":board,"bomb_blast_strength":bomb_blast_strength,"bomb_life":bomb_life,"flame_life":flame_life,
                            "position":position,"ammo":ammo,"blast_strength":blast_strength})

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


if __name__ == '__main__':
    env = PomFFA()
    obs = env.reset()
    print(obs)

    # for i in range(20):
    #     obs, reward, done, _ = env.step(0)
    #     print(obs, reward, done)
