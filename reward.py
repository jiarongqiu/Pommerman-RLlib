import math
from pommerman import constants
from utils import compute_wood

class Reward():

    def __init__(self, config=None):
        self.config = config
        print("Reward Config:", config)

    def get_reward(self, action,obs,info):
        if not self.config or self.config['version'] == 'v0':
            return self.v0(action,obs,info)
        elif self.config['version'] == 'v1':
            return self.v1(action,obs,info)
        elif self.config['version'] == 'v2':
            return self.v2(action, obs, info)
        elif self.config['version'] == 'v3':
            return self.v3(action, obs, info)
        ### Add more reward funcs here

        return 0

    def v0(self, action,obs,info):
        agent_id = info['agent_id']
        if len(obs["alive"]) == 1:
            if agent_id in obs['alive']:
                reward = 1
            else:
                reward = -1
        elif obs['step_count'] >= 800:
            reward = -1
        else:
            # Game running: 0 for alive, -1 for dead.
            if agent_id in obs['alive']:
                reward = 0
            else:
                reward = -1
        return reward,info

    def v1(self, action, obs, state):
        #min -2 max 2
        agent_id = state['agent_id']
        reward = 0

        if obs['position'] not in state['visited']:
            state['visited'].add(obs['position'])
            reward += 0.001

        if obs['blast_strength']>state['strength']:
            reward += 0.01
            state['strength'] = obs['blast_strength']

        if obs['ammo']>state['ammo']:
            reward += 0.01
            state['ammo'] = obs['ammo']

        if len(obs['alive']) < len(state['alive']):
            if agent_id in obs['alive']:
                reward += 0.5
            else:
                reward += -2
            state['alive'] = obs['alive']
        elif obs['step_count'] >= 800:
            reward += -2

        return reward, state

    def v2(self, action, obs, state):
        #min -2 max 2
        agent_id = state['agent_id']
        reward = 0
        if action == 5 and state["prev_obs"]["ammo"]>0:
            if obs['position'] not in state["bombs"]:
                state["bombs"][obs['position']] = 0

        if obs['position'] not in state['visited']:
            state['visited'].add(obs['position'])
            reward += 0.01

        # if obs['blast_strength']>state['strength']:
        #     reward += 0.01
        #     state['strength'] = obs['blast_strength']

        # if obs['ammo']>state['ammo']:
        #     reward += 0.01
        #     state['ammo'] = obs['ammo']

        if len(obs['alive']) < len(state['alive']):
            if agent_id in obs['alive']:
                reward += 1
            else:
                reward += -1
            state['alive'] = obs['alive']
        else:
            reward += -1.0/800
        # elif (obs['step_count']+1)%200 == 0:
        #     reward += -0.25
        delete = set()
        for y,x in state["bombs"]:
            neighbor = compute_wood(obs['board'], (y, x), obs['blast_strength'])
            wood_num = len([e for e in neighbor if e == 2])
            if obs["board"][y][x] == 3 or obs["board"][y][x] == state["agent_id"]:
                if wood_num == 0:
                    delete.add((y,x))
                else:
                    state["bombs"][y,x] = wood_num
            else:
                diff = state["bombs"][y,x] - wood_num
                reward += 0.1*diff
                delete.add((y,x))
        state["bombs"] = {(y,x):state["bombs"][y,x] for y,x in state["bombs"] if (y,x) not in delete}

        return reward, state

    def v3(self, action, obs, state):
        agent_id = state['agent_id']
        reward = 0
        if action == 5 and state["prev_obs"]["ammo"]>0:
            if obs['position'] not in state["bombs"]:
                state["bombs"][obs['position']] = 0
                wood_num = compute_wood(obs['board'], obs['position'], obs['blast_strength'])
                reward += 0.1*wood_num

        if obs['position'] not in state['visited']: # +1
            state['visited'].add(obs['position'])
            reward += 0.1

        if len(obs['alive']) < len(state['alive']):
            if agent_id in obs['alive']:
                reward += 1
            state['alive'] = obs['alive']
        if obs['blast_strength']>state['blast_strength']:
            reward += 0.1
            state['blast_strength'] = obs['blast_strength']
        if obs['ammo'] > state['ammo']:
            reward += 0.1
            state['ammo'] = obs['ammo']
        # else:
        #     reward += -0.5/800
        # # elif (obs['step_count']+1)%200 == 0:
        # #     reward += -0.25
        delete = set()
        for y,x in state["bombs"]:
            wood_num = compute_wood(obs['board'], (y, x), obs['blast_strength'])
            if obs["board"][y][x] == 3 or obs["board"][y][x] == state["agent_id"]:
                if wood_num == 0:
                    delete.add((y,x))
                else:
                    state["bombs"][y,x] = wood_num
            else:
                diff = state["bombs"][y,x] - wood_num
                reward += 0.1*diff
                delete.add((y,x))
        state["bombs"] = {(y,x):state["bombs"][y,x] for y,x in state["bombs"] if (y,x) not in delete}

        return reward, state