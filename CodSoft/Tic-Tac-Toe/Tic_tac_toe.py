            return True
    for col in range(3):
        if all([board[row][col] == player for row in range(3)]):
            return True
    if all([board[i][i] == player for i in range(3)]) or all([board[i][2-i] == player for i in range(3)]):
        return True
    return False

def check_full(board):
    return all([cell != " " for row in board for cell in row])

def get_available_moves(board):
    moves = []
    for i in range(3):
        for j in range(3):
            if board[i][j] == " ":
                moves.append((i, j))
    return moves

def minimax(board, depth, is_maximizing, alpha=-math.inf, beta=math.inf, use_alpha_beta=False):
    if check_winner(board, "O"):
        return 1
    if check_winner(board, "X"):
        return -1
    if check_full(board):
        return 0

    if is_maximizing: