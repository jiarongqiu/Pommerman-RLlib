class Reward():

    def __init__(self, config):
        self.config = config

    def get_reward(self, obs, action, agent_id):
        if not self.config or self.config['version'] == 'default':
            return self.default(obs, action, agent_id)

        ### Add more reward funcs here

        return 0

    def default(self, obs, action, agent_id):
        if len(obs["alive"]) == 1:
            # An agent won. Give them +1, others -1.
            if agent_id in obs['alive']:
                return 1
            else:
                return -1

        elif obs["step_count"] >= 500:
            # Game is over from time. Everyone gets -1.
            return -1
        else:
            # Game running: 0 for alive, -1 for dead.
            if agent_id in obs['alive']:
                return 0
            else:
                return -1
