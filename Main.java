import java.util.Scanner;

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
                int person = -1;
                int computer = 1;
                f = true;
                System.out.println("You are black");
            } else if (first.equals("N")) {
                int person = 1;
                int computer = -1;
                f = true;
                System.out.println("You are white");
            } else {
                System.out.println("Invalid input");
            }
        }
        int turn = -1;
        int total_pieces = 0;
        int black_pieces = 2;
        int white_pieces = 2;
        Board board = new Board(8);
        board.initializeBoard();
        board.printBoard();
        Move move = new Move();
        boolean end = false;
        while (end == false) {
            boolean rightmove = false;
            while (rightmove == false) {
                System.out.println("Give row");
                int row = in.nextInt();
                System.out.println("Give col");
                int col = in.nextInt();
                Move playerMove = new Move(row, col, 0);
                if (playerMove.isValid(board, turn, playerMove)) {
                    board.setLastMove(playerMove, turn);
                    board.makeMove(playerMove, turn);
                    board.printBoard();
                    turn = turn * (-1);
                    rightmove = true;
                    total_pieces++;
                    black_pieces = board.countBlack();
                    white_pieces = board.countWhite();
                } else {
                    System.out.println("Invalid move\n");
                    board.printBoard();
                }
            }
        }

        /*
         * System.out.println("Give row");
         * row = in.nextInt();
         * System.out.println("Give col");
         * col = in.nextInt();
         * playerMove = new Move(row, col, 0);
         * board.setLastMove(playerMove, turn % 2);
         * board.printBoard();
         */
    }

}
