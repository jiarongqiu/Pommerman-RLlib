

def compute_wood(board, pos, radius,rigid=1,wood=2):
    n = board.shape[0]
    y,x = pos
    incres = [(1,0),(-1,0),(0,1),(0,-1)]
    res = 0
    for dy,dx in incres:
        y2,x2 = y,x
        for _ in range(1,radius):
            y2 += dy
            x2 += dx
            if 0 <= x2 < n and 0 <= y2 <n:
                if board[y2][x2] == wood:
                    res += 1
                    break
                elif board[y2][x2] == rigid:
                    break
            else:
                break
    return res


def create_cross(board, pos, radius):
    n = board.shape[0]
    y,x=pos
    board[y][x] = 1
    radius = int(radius)
    for l in range(1,radius):
        y1, y2 = y - l, y + l
        x1, x2 = x - l, x + l
        if y1 >= 0:
            board[y1][x] = 1
        if y2 < n:
            board[y2][x] = 1
        if x1 >= 0:
            board[y][x1] = 1
        if x2 < n:
            board[y][x2] = 1
    return board