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
        Board board = new Board(8);
        board.initializeBoard();
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
                    Move playerMove = new Move(row, col);
                    board.makeMove(playerMove.getRow(), playerMove.getCol(), Board.B);
                    break;
                case Board.B:
                    System.out.println("White plays");
                    System.out.println("Give row");
                    row = in.nextInt();
                    System.out.println("Give col");
                    col = in.nextInt();
                    playerMove = new Move(row, col);
                    board.makeMove(playerMove.getRow(), playerMove.getCol(), Board.W);
                default:
                    break;
            }
        }

    }

}
