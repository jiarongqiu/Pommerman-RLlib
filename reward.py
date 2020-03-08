from pommerman import constants


class Reward():

    def __init__(self, config=None):
        self.config = config
        print("Reward Config:", config)

    def get_reward(self, obs, action, agent_id):
        if not self.config or self.config['version'] == 'default':
            return self.default(obs, action, agent_id)
        elif self.config['version'] == 'v0':
            return self.v0(obs, action, agent_id)

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

    def v0(self, obs, action, agent_id):

        reward = 0
        x, y = obs['position']
        board = obs['board']
        alive_agents = obs['alive']
        blast_strength = obs['blast_strength']
        ammo = obs['ammo']
        board_size = board.shape[0]


        if 1 <= action <= 4:
            if action == 1:
                y -= 1
            elif action == 2:
                y += 1
            elif action == 3:
                x -= 1
            else:
                x += 1
            if 0<=x<board_size and 0<=y<board_size:
                if board[y][x] == 6 or board[y][x] == 7:
                    reward += 5
            else:
                reward -= 5

        if action == 5:
            def check_valid_bomb(x, y, board, blast_strength):
                x_min = max(0, x - blast_strength)
                x_max = min(board_size, x + blast_strength)
                for j in range(x_min, x_max):
                    if board[y][j] == 2:
                        return True

                y_min = max(0, y - blast_strength)
                y_max = min(board_size, y + blast_strength)
                for i in range(y_min, y_max):
                    if board[i][x] == 2:
                        return True
            if check_valid_bomb(x,y,board,blast_strength):
                reward += 2

        if len(alive_agents) == 1 and agent_id in alive_agents:
            reward += 100

        if agent_id not in alive_agents:
            reward -= 100

        return reward