import java.util.ArrayList;

class Board {
    public static final int W = 1;
    public static final int B = -1;
    public static final int EMPTY = 0;

    private int[][] gameBoard;

    private int lastPlayer;

    private Move lastMove;

    private int dimension;

    public Board() {
    }

    public Board(int dimension) {
        this.dimension = dimension;
        this.gameBoard = new int[dimension][dimension];
        for (int i = 0; i < dimension; i++) {
            for (int j = 0; j < dimension; j++) {
                gameBoard[i][j] = EMPTY;
            }
        }

    }

    // copy constructor
    public Board(Board board) {
        // this.dimension = dimension;
        this.gameBoard = new int[dimension][dimension];
        for (int i = 0; i < dimension; i++) {
            for (int j = 0; j < dimension; j++) {
                gameBoard[i][j] = this.gameBoard[i][j] = board.gameBoard[i][j];
                ;
            }
        }
    }

    public void initializeBoard() {
        gameBoard[3][3] = W;
        gameBoard[4][4] = W;
        gameBoard[3][4] = B;
        gameBoard[4][3] = B;
    }

    public void printBoard() {
        for (int i = 0; i < dimension; i++) {
            for (int j = 0; j < dimension; j++) {
                if (gameBoard[i][j] == W) {
                    System.out.print("W");
                } else if (gameBoard[i][j] == B) {
                    System.out.print("B");
                } else {
                    System.out.print("-");
                }

                if (j == dimension - 1) {
                    System.out.println();
                }
            }
        }
    }

    public void print() {
    }

    ArrayList<Board> getChildren(int letter) {
        return null;
    }

    public int evaluate() {
        return 0;
    }

    public boolean isTerminal() {
        return true;
    }

    public Move getLastMove() {
        return this.lastMove;
    }

    public int getLastPlayer() {
        return this.lastPlayer;
    }

    public int[][] getGameBoard() {
        return this.gameBoard;
    }

    public int getDimension() {
        return this.dimension;
    }

    void setGameBoard(int[][] gameBoard) {
        for (int i = 0; i < this.dimension; i++) {
            for (int j = 0; j < this.dimension; j++) {
                this.gameBoard[i][j] = gameBoard[i][j];
            }
        }
    }

    void setLastMove(Move lastMove) {
        this.lastMove.setRow(lastMove.getRow());
        this.lastMove.setCol(lastMove.getCol());
        this.lastMove.setValue(lastMove.getValue());
    }

    void setLastPlayer(int lastPlayer) {
        this.lastPlayer = lastPlayer;
    }

}