from pommerman.agents import BaseAgent


class HoldAgent(BaseAgent):

    def act(self, obs, action_space):
        return 0


class HoldRandomAgent(BaseAgent):

    def act(self, obs, action_space):
        action = action_space.sample()
        if action == 5:
            action = 0
        return action
