import java.util.ArrayList;
import java.util.Random;

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
        this.maxDepth = 0;
    }

    public Move MiniMax(Board board) {
        if (this.playerColour == Board.W) {
            return max(new Board(board), 0, Integer.MIN_VALUE, Integer.MAX_VALUE);
        } else {
            return min(new Board(board), 0, Integer.MIN_VALUE, Integer.MAX_VALUE);
        }
    }

    public Move max(Board board, int depth, int alpha, int beta) {
        Random r = new Random();
        if (board.isTerminal() || (depth == this.maxDepth)) { // elegxos kathe katastashs paidiou mexri na anagkastei na
                                                              // termatisei
            return new Move(board.getLastMove().getRow(), board.getLastMove().getCol(), board.evaluate());
        }
        ArrayList<Board> children = board.getChildren(Board.W);
        Move maxMove = new Move(Integer.MIN_VALUE);
        for (Board child : children) {
            Move move = min(child, depth + 1, alpha, beta);
            if (move.getValue() >= maxMove.getValue()) { // epilogh kathe fora tou pio symferontos value edw max
                if ((move.getValue()) == maxMove.getValue()) {
                    if (r.nextInt(2) == 0) { // an dyo kinhseis exoun idio value epelkse mia tyxaia anti gia thn prwth
                        maxMove.setRow(child.getLastMove().getRow());
                        maxMove.setCol(child.getLastMove().getCol());
                        maxMove.setValue(move.getValue());
                    }
                } else {
                    maxMove.setRow(child.getLastMove().getRow());
                    maxMove.setCol(child.getLastMove().getCol());
                    maxMove.setValue(move.getValue());
                }
                alpha = Math.max(alpha, maxMove.getValue()); // ypologismos timhs a kai sygkrish me b
                if (alpha >= beta) {
                    break;
                }
            }
        }
        return maxMove;

    }

    public Move min(Board board, int depth, int alpha, int beta) {
        Random r = new Random();
        if (board.isTerminal() || (depth == this.maxDepth)) {
            return new Move(board.getLastMove().getRow(), board.getLastMove().getCol(), board.evaluate());
        }
        ArrayList<Board> children = board.getChildren(Board.B);
        Move minMove = new Move(Integer.MAX_VALUE);
        for (Board child : children) {
            Move move = max(child, depth + 1, alpha, beta);
            if (move.getValue() <= minMove.getValue()) { // epilogh kathe fora tou pio symferontos value edw min
                if ((move.getValue()) == minMove.getValue()) {
                    if (r.nextInt(2) == 0) {
                        minMove.setRow(child.getLastMove().getRow());
                        minMove.setCol(child.getLastMove().getCol());
                        minMove.setValue(move.getValue());
                    }
                } else {
                    minMove.setRow(child.getLastMove().getRow());
                    minMove.setCol(child.getLastMove().getCol());
                    minMove.setValue(move.getValue());
                }
                beta = Math.min(beta, minMove.getValue()); // ypologismos timhs b kai sygkrish me a
                if (alpha >= beta) {
                    break; // Alpha cutoff
                }
            }
        }
        return minMove;
    }

}