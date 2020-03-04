import gym
import pommerman
import numpy as np
from pommerman.envs.v0 import Pomme
from pommerman import agents,constants
from gym.envs.registration import register
from gym import spaces


# register(
#     id='PomFFA-v2',
#     entry_point='env:PomFFA',
# )

class PomFFA(gym.Env):
    '''
    A wrapped Pommerman v0 environment for usage with Ray RLlib. The v0 environment is the base environment used in
    the NIPS'18 competition. Contrary to v1 it doesn't collapse walls and also doesn't allow for radio communication
    between agents (as does v2).
    Agents are identified by (string) agent IDs: `AGENT_IDS`
    (Note that these "agents" here are not to be confused with RLlib agents.)
    '''

    def __init__(self, env_config=None):

        pomme_config = pommerman.configs.ffa_competition_env()

        if env_config:
            for k, v in env_config.items():
                pomme_config['env_kwargs'][k] = v
        print(pomme_config['env_kwargs'])
        self.pomme = Pomme(**pomme_config['env_kwargs'])
        self.observation_space = self.init_observation_space(pomme_config['env_kwargs'])
        self.action_space = self.pomme.action_space
        print(self.observation_space.shape)
        agent_list = [agents.SimpleAgent(), agents.SimpleAgent(), agents.SimpleAgent(), agents.RandomAgent()]
        self.init(agent_list, pomme_config)

        # record all agents
        self.all_obs = None
        self.all_action = None
        self.alive_agents = [10,11,12,13]

        self.training_agent_id = 13


    def init(self, agent_list, pomm_config):
        for id_, agent in enumerate(agent_list):
            assert isinstance(agent, agents.BaseAgent)
            agent.init_agent(id_, pomm_config['game_type'])
        self.pomme.set_agents(agent_list)
        self.pomme.set_init_game_state(None)

    def reset(self):
        obs = self.pomme.reset()
        self.all_obs = obs
        obs = self.get_for_training_agent(obs)
        # obs = self.obs2vec(obs)
        obs = self.remove_useless_obs(obs)
        return obs

    def step(self, action):
        actions = self.pomme.act(self.all_obs)
        if self.alive_agents and 13 in self.alive_agents:
            actions = self.set_for_training_agent(actions, action)
        else:
            actions = self.set_for_training_agent(actions, 0)
        print(actions)
        obs, rewards, done, info = self.pomme.step(actions)
        self.all_obs = obs
        obs = self.get_for_training_agent(obs)
        reward = self.get_for_training_agent(rewards)
        self.alive_agents = obs['alive']
        # obs = self.obs2vec(obs)
        obs = self.remove_useless_obs(obs)
        return obs, reward, done, {}

    def get_for_training_agent(self, inputs):
        order = self.training_agent_id - 10
        return inputs[order]

    def set_for_training_agent(self, inputs, value):
        order = self.training_agent_id - 10
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
        bss = board_size ** 2

        board = spaces.Box(low=0,high=len(constants.Item),shape=(board_size,board_size))
        bomb_blast_strength = spaces.Box(low=0,high=num_items,shape=(board_size,board_size))
        bomb_life = spaces.Box(low=0,high=9,shape=(board_size,board_size))
        flame_life = spaces.Box(low=0,high=3,shape=(board_size,board_size))
        position = spaces.Box(low=0,high=board_size,shape=(2,))
        blast_strength = spaces.Box(low=1,high=num_items,shape=(1,))
        ammo = spaces.Box(low=1,high=num_items,shape=(1,))

        # min_obs = [0] * 4 * bss + [0] * 4
        # max_obs = [len(constants.Item)] * bss + [board_size] * bss + [25] * bss + [5]*bss
        # max_obs += [board_size] * 2 + [num_items] * 2
        # return spaces.Box(np.array(min_obs), np.array(max_obs))

        return spaces.Dict({"board":board,"bomb_blast_strength":bomb_blast_strength,"bomb_life":bomb_life,"flame_life":flame_life,
                            "position":position,"ammo":ammo,"blast_strength":blast_strength})

    @staticmethod
    def obs2vec(obs):
        board = obs['board'].flatten()
        bomb_blast_strength = obs['bomb_blast_strength'].flatten()
        bomb_life = obs['bomb_life'].flatten()
        flame_life = obs['flame_life'].flatten()
        position = obs['position']
        blast_strength = obs['blast_strength']
        ammo = obs['ammo']
        res = np.hstack([board,bomb_blast_strength,bomb_life,flame_life]+list(position) + [blast_strength,ammo])
        return res

    @staticmethod
    def remove_useless_obs(obs):
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
        # del obs['ammo']
        # del obs['blast_strength']
        return obs

    def render(self):
        self.pomme.render()


if __name__ == '__main__':
    # env = gym.make("PomFFA-v0")
    env = PomFFA()
    obs = env.reset()
    print (obs)
    # for i in range(20):
    #     obs, reward, done, _ = env.step(0)
    #     print(obs, reward, done)
