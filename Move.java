public class Move {
    private int row;
    private int col;
    private int value;

    Move() {
        this.row = -1;
        this.col = -1;
        this.value = 0;
    }

    Move(int row, int col) {
        this.row = row;
        this.col = col;
        this.value = -1;
    }

    Move(int value) {
        this.row = -1;
        this.col = -1;
        this.value = value;
    }

    Move(int row, int col, int value) {
        this.row = row;
        this.col = col;
        this.value = value;
    }

    int getRow() {
        return this.row;
    }

    int getCol() {
        return this.col;
    }

    int getValue() {
        return this.value;
    }

    void setRow(int row) {
        this.row = row;
    }

    void setCol(int col) {
        this.col = col;
    }

    void setValue(int value) {
        this.value = value;
    }

    boolean isValid(Board b, int colour, Move move) {
        int gb[][] = b.getGameBoard();
        int row = move.getRow();
        int col = move.getCol();
        if (row < 0 || row > b.getDimension() || col < 0 || col > b.getDimension() || gb[row][col] != 0) {
            return false;
        }
        int enemycolour = (colour == 1) ? -1 : 1;
        int dir[][] = { { 0, 1 }, { 1, 1 }, { 1, 0 }, { -1, 0 }, { 0, -1 }, { -1, -1 }, { 1, -1 }, { -1, 1 } };
        for (int[] d : dir) {
            int temp_r = row + d[0];
            int temp_c = col + d[1];
            boolean exists_reverse = false;
            while (temp_r > 0 && temp_r < b.getDimension() && temp_c > 0 && temp_c < b.getDimension()) {
                if (gb[temp_r][temp_c] == enemycolour) {
                    exists_reverse = true;
                } else if (gb[temp_r][temp_c] != 0 && gb[temp_r][temp_c] != enemycolour && exists_reverse == true) {
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
}
