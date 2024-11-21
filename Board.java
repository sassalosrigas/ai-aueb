import java.util.ArrayList;

class Board {
    public static final int W = 1;
    public static final int B = -1;
    public static final int EMPTY = 0;
    public static final int dimension = 8;

    private int[][] gameBoard;

    private int lastPlayer;

    private Move lastMove;

    public Board() {
        this.lastMove = new Move();
        this.lastPlayer = W;
        this.gameBoard = new int[8][8];
        for (int i = 0; i < dimension; i++) {
            for (int j = 0; j < dimension; j++) {
                this.gameBoard[i][j] = EMPTY;
            }
        }
        gameBoard[3][3] = B;
        gameBoard[4][4] = B;
        gameBoard[3][4] = W;
        gameBoard[4][3] = W;
        this.setLastPlayer(W);
    }

    // copy constructor
    public Board(Board board) {
        this.lastMove = board.lastMove;
        this.lastPlayer = board.lastPlayer;
        this.gameBoard = new int[8][8];
        for (int i = 0; i < dimension; i++) {
            for (int j = 0; j < dimension; j++) {
                this.gameBoard[i][j] = board.gameBoard[i][j];
            }
        }
    }

    public void printBoard() {
        for (int i = 0; i < 8; i++) {
            System.out.print(i + 1);
            for (int j = 0; j < 8; j++) {
                if (gameBoard[i][j] == W) {
                    System.out.print("W");
                } else if (gameBoard[i][j] == B) {
                    System.out.print("B");
                } else {
                    System.out.print("-");
                }

                if (j == 7) {
                    System.out.println();
                }
            }
        }
        for (int i = 0; i <= dimension; i++) {
            System.out.print((i == 0) ? " " : i);
        }
        System.out.println();
    }

    ArrayList<Board> getChildren(int colour) {
        ArrayList<Board> children = new ArrayList<>();
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                if (this.isValid(i, j, colour)) {
                    Board child = new Board(this);
                    child.makeMove(i, j, colour);
                    children.add(child);
                }
            }
        }
        return children;
    }

    public int evaluate() {
        return this.countBlack() - this.countWhite();
    }

    public boolean isValid(int row, int col, int colour) {
        int gb[][] = this.getGameBoard();
        if (row < 0 || row >= dimension || col < 0 || col >= dimension || gb[row][col] != EMPTY) {
            return false;
        }
        int enemycolour = (colour == 1) ? -1 : 1;
        int dir[][] = { { 0, 1 }, { 1, 1 }, { 1, 0 }, { -1, 0 }, { 0, -1 }, { -1, -1 }, { 1, -1 }, { -1, 1 } };
        for (int[] d : dir) {
            int temp_r = row + d[0];
            int temp_c = col + d[1];
            boolean exists_reverse = false;
            while (temp_r >= 0 && temp_r < dimension && temp_c >= 0 && temp_c < dimension) {
                if (gb[temp_r][temp_c] == enemycolour) {
                    exists_reverse = true;
                } else if (gb[temp_r][temp_c] != EMPTY && gb[temp_r][temp_c] != enemycolour && exists_reverse == true) {
                    return true;
                } else {
                    break;
                }
                temp_r = temp_r + d[0];
                temp_c = temp_c + d[1];
            }
        }
        return false;
    }

    public boolean isTerminal() {
        if (this.count_pieces() == (64)) {
            if (this.countBlack() > countWhite()) {
                System.out.println("Black wins with");
            } else if (this.countWhite() > this.countBlack()) {
                System.out.println("White wins");
            } else {
                System.out.println("It's a tie");
            }
            System.out.println("White pieces " + this.countWhite());
            System.out.println("Black pieces " + this.countBlack());
            return true;
        }
        int colour = this.getLastPlayer() * (-1);
        for (int i = 0; i < dimension; i++) {
            for (int j = 0; j < dimension; j++) {
                if (this.gameBoard[i][j] == EMPTY) {
                    Move move = new Move(i, j);
                    if (this.isValid(move.getRow(), move.getCol(), colour)) {
                        return false;
                    }
                }
            }
        }
        if (colour == this.W) {
            System.out.println("No more valid moves for white swithing to black");
        } else {
            System.out.println("No more valid moves for black swithing to white");
        }
        this.setLastPlayer(colour);
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                if (this.gameBoard[i][j] == EMPTY) {
                    Move move = new Move(i, j);
                    if (this.isValid(move.getRow(), move.getCol(), colour * (-1))) {
                        return false;
                    }
                }
            }
        }
        System.out.println("No more valid moves game over");
        if (this.countBlack() > countWhite()) {
            System.out.println("Black wins");
        } else if (this.countWhite() > this.countBlack()) {
            System.out.println("White wins");
        } else {
            System.out.println("It's a tie");
        }
        System.out.println("White pieces " + this.countWhite());
        System.out.println("Black pieces " + this.countBlack());
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

    public void setGameBoard(int[][] gameBoard) {
        for (int i = 0; i < dimension; i++) {
            for (int j = 0; j < dimension; j++) {
                this.gameBoard[i][j] = gameBoard[i][j];
            }
        }
    }

    public void setLastMove(Move lastMove) {
        this.lastMove.setRow(lastMove.getRow());
        this.lastMove.setCol(lastMove.getCol());
        this.lastMove.setValue(lastMove.getValue());
    }

    public void makeMove(int row, int col, int colour) {
        int enemycolour = (colour == 1) ? -1 : 1;
        if (!isValid(row, col, colour)) {
            return;
        }
        int dir[][] = { { 0, 1 }, { 1, 1 }, { 1, 0 }, { -1, 0 }, { 0, -1 }, { -1, -1 }, { 1, -1 }, { -1, 1 } };
        this.gameBoard[row][col] = colour;
        for (int[] d : dir) {
            int temp_r = row + d[0];
            int temp_c = col + d[1];
            boolean exists_reverse = false;
            while (temp_r >= 0 && temp_r < dimension && temp_c >= 0 && temp_c < dimension) {
                if (this.gameBoard[temp_r][temp_c] == enemycolour) {
                    exists_reverse = true;
                } else if (this.gameBoard[temp_r][temp_c] == colour && exists_reverse == true) {
                    temp_r = row + d[0];
                    temp_c = col + d[1];
                    while (temp_r >= 0 && temp_r < dimension && temp_c >= 0 && temp_c < dimension) {
                        if (this.gameBoard[temp_r][temp_c] == EMPTY || this.gameBoard[temp_r][temp_c] != enemycolour) {
                            break;
                        }
                        if (this.gameBoard[temp_r][temp_c] == enemycolour) {
                            this.gameBoard[temp_r][temp_c] = colour;
                        }
                        temp_r += d[0];
                        temp_c += d[1];
                    }
                    break;
                } else {
                    break;
                }
                temp_r = temp_r + d[0];
                temp_c = temp_c + d[1];
            }
        }
        this.lastMove = new Move(row, col, this.evaluate());
        this.setLastPlayer(this.getLastPlayer() * (-1));
    }

    public void setLastPlayer(int lastPlayer) {
        this.lastPlayer = lastPlayer;
    }

    public int countWhite() {
        int count = 0;
        for (int i = 0; i < dimension; i++) {
            for (int j = 0; j < dimension; j++) {
                if (this.gameBoard[i][j] == W) {
                    count++;
                }
            }
        }
        return count;
    }

    public int countBlack() {
        int count = 0;
        for (int i = 0; i < dimension; i++) {
            for (int j = 0; j < dimension; j++) {
                if (this.gameBoard[i][j] == B) {
                    count++;
                }
            }
        }
        return count;
    }

    public int count_pieces() {
        return this.countWhite() + this.countBlack();
    }

}