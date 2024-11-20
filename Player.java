class Player {
    private int maxDepth;
    private int playerColour;

    public Player() {
    }

    public Player(int maxDepth, int playerColour) {
        this.maxDepth = maxDepth;
        this.playerColour = playerColour;
    }

    public Player(int playerColour) {
        this.playerColour = playerColour;
    }

    public Move MiniMax(Board board) {
        if (this.playerColour == Board.W) {
            return max(new Board(board), this.maxDepth);
        } else {
            return min(new Board(board), this.maxDepth);
        }
    }

    public Move max(Board board, int depth) {
        return null;
    }

    public Move min(Board board, int depth) {
        return null;
    }
}