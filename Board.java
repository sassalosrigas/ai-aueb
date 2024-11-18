import java.util.ArrayList;

class Board {
    public static final int W = 1;
    public static final int B = -1;
    public static final int EMPTY = 0;

    private int[][] gameBoard;

    private int lastPlayer;

    private Move lastMove;

    private int dimension;

    private int lastcolor;

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
        gameBoard[3][3] = B;
        gameBoard[4][4] = B;
        gameBoard[3][4] = W;
        gameBoard[4][3] = W;
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
        System.out.println();
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

    void setLastMove(Move lastMove, int color) {
        if (this.lastMove == null) {
            this.lastMove = new Move();
        }
        this.lastMove.setRow(lastMove.getRow());
        this.lastMove.setCol(lastMove.getCol());
        this.lastMove.setValue(lastMove.getValue());
        this.lastcolor = color;
        /*
         * if (color == 1) {
         * this.gameBoard[this.lastMove.getRow()][this.lastMove.getCol()] = W;
         * } else {
         * this.gameBoard[this.lastMove.getRow()][this.lastMove.getCol()] = B;
         * }
         */

    }

    void makeMove(Move lastMove, int colour) {
        int row = this.lastMove.getRow();
        int col = this.lastMove.getCol();
        int enemycolour = (colour == 1) ? -1 : 1;
        int dir[][] = { { 0, 1 }, { 1, 1 }, { 1, 0 }, { -1, 0 }, { 0, -1 }, { -1, -1 }, { 1, -1 }, { -1, 1 } };
        this.gameBoard[row][col] = colour;
        for (int[] d : dir) {
            int temp_r = row + d[0];
            int temp_c = col + d[1];
            boolean exists_reverse = false;
            while (temp_r > 0 && temp_r < this.getDimension() && temp_c > 0 && temp_c < this.getDimension()) {
                if (this.gameBoard[temp_r][temp_c] == enemycolour) {
                    exists_reverse = true;
                } else if (this.gameBoard[temp_r][temp_c] == enemycolour * (-1) && exists_reverse == true) {
                    temp_r = row + d[0];
                    temp_c = col + d[1];
                    while (temp_r > 0 && temp_r < this.getDimension() && temp_c > 0 && temp_c < this.getDimension()) {
                        if (this.gameBoard[temp_r][temp_c] == enemycolour) {
                            this.gameBoard[temp_r][temp_c] = this.gameBoard[temp_r][temp_c] * (-1);
                        }
                        temp_r = temp_r + d[0];
                        temp_c = temp_c + d[1];
                    }
                    break;
                } else {
                    break;
                }
                temp_r = temp_r + d[0];
                temp_c = temp_c + d[1];
            }
        }
    }

    void setLastPlayer(int lastPlayer) {
        this.lastPlayer = lastPlayer;
    }

    int countWhite() {
        int count = 0;
        for (int i = 0; i < this.getDimension(); i++) {
            for (int j = 0; j < this.getDimension(); j++) {
                if (this.gameBoard[i][j] == W) {
                    count++;
                }
            }
        }
        return count;
    }

    int countBlack() {
        int count = 0;
        for (int i = 0; i < this.getDimension(); i++) {
            for (int j = 0; j < this.getDimension(); j++) {
                if (this.gameBoard[i][j] == B) {
                    count++;
                }
            }
        }
        return count;
    }

}