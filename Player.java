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
            return max(new Board(board), 0);
        } else {
            return min(new Board(board), 0);
        }
    }

    public Move max(Board board, int depth) {
        Random r = new Random();
        /*
         * If MAX is called on a state that is terminal or after a maximum depth is
         * reached,
         * then a heuristic is calculated on the state and the move returned.
         */
        if (board.isTerminal() || (depth == this.maxDepth)) {
            return new Move(board.getLastMove().getRow(), board.getLastMove().getCol(), board.evaluate());
        }
        // The children-moves of the state are calculated
        ArrayList<Board> children = board.getChildren(Board.W);
        Move maxMove = new Move(Integer.MIN_VALUE); // put max node initially to smallest value.
        for (Board child : children) {
            // And for each child min is called, on a lower depth
            Move move = min(child, depth + 1);
            // The child-move with the greatest value is selected and returned by max
            if (move.getValue() >= maxMove.getValue()) {
                // If the heuristic has the save value then we randomly choose one of the two
                // moves
                if ((move.getValue()) == maxMove.getValue()) {
                    if (r.nextInt(2) == 0) {
                        maxMove.setRow(child.getLastMove().getRow());
                        maxMove.setCol(child.getLastMove().getCol());
                        maxMove.setValue(move.getValue());
                    }
                } else {
                    maxMove.setRow(child.getLastMove().getRow());
                    maxMove.setCol(child.getLastMove().getCol());
                    maxMove.setValue(move.getValue());
                }
            }
        }
        return maxMove;

    }

    public Move min(Board board, int depth) {
        Random r = new Random();
        if (board.isTerminal() || (depth == this.maxDepth)) {
            return new Move(board.getLastMove().getRow(), board.getLastMove().getCol(), board.evaluate());
        }
        ArrayList<Board> children = board.getChildren(Board.B);
        Move minMove = new Move(Integer.MAX_VALUE);
        for (Board child : children) {
            Move move = max(child, depth + 1);
            if (move.getValue() <= minMove.getValue()) {
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
            }
        }
        return minMove;
    }
}