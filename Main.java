import java.util.Scanner;
import java.util.ArrayList;;

class Main {

    public static void main(String args[]) {
        Scanner in = new Scanner(System.in);
        System.out.println("Give depth");
        int depth = in.nextInt();
        in.nextLine();
        boolean f = false;
        while (f == false) {
            System.out.println("Do you want to play first? Y(Yes) N(No)");
            String first = in.nextLine();
            if (first.equals("Y")) {
                // Player playerBlack = new Player(depth, Board.B);
                // Player playerWhite = new Player(depth, Board.W);
                int computer = 1;
                f = true;
                System.out.println("You are black");
            } else if (first.equals("N")) {
                // Player playerBlack = new Player(depth, Board.B);
                // Player playerWhite = new Player(depth, Board.W);
                int computer = -1;
                f = true;
                System.out.println("You are white");
            } else {
                System.out.println("Invalid input");
            }
        }
        Player playerBlack = new Player(depth, Board.B);
        Player playerWhite = new Player(depth, Board.W);
        Board board = new Board();
        board.printBoard();
        Move move = new Move();
        while (!board.isTerminal()) {
            switch (board.getLastPlayer()) {
                case Board.W:
                    System.out.println("Black plays");
                    System.out.println("Give row");
                    int row = in.nextInt();
                    System.out.println("Give col");
                    int col = in.nextInt();
                    Move playerMove = new Move(row, col, board.evaluate());
                    board.makeMove(playerMove.getRow(), playerMove.getCol(), Board.B);
                    break;
                case Board.B:
                    System.out.println("White plays");
                    Move moveW = playerWhite.MiniMax(board);
                    board.makeMove(moveW.getRow(), moveW.getCol(), Board.W);
                    // System.out.println("Give row");
                    // row = in.nextInt();
                    // System.out.println("Give col");
                    // col = in.nextInt();
                    // playerMove = new Move(row, col);
                    // board.makeMove(playerMove.getRow(), playerMove.getCol(), Board.W);
                    // board.getChildren(Board.B);
                    break;
                default:
                    break;
            }
            board.printBoard();
            // System.out.println("Position evaluation " + board.evaluate());
        }
    }
}
