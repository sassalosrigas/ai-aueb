import java.util.Scanner;

class Main {

    public static void main(String args[]) {
        Scanner in = new Scanner(System.in);
        System.out.println("Give depth");
        int depth = in.nextInt();
        boolean f = false;
        while (f == false) {
            System.out.println("Do you want to play first? Y(Yes) N(No)");
            String first = in.nextLine();
            if (first.equals("Y")) {
                int person = 1;
                f = true;
                System.out.println("You are white");
            } else if (first.equals("N")) {
                int person = 2;
                f = true;
                System.out.println("You are black");
            } else {
                System.out.println("Invalid input");
            }
        }
        int turn = 1;
        Board board = new Board(8);
        board.initializeBoard();
        board.printBoard();
        Move move = new Move();
        System.out.println("Give row");
        int row = in.nextInt();
        System.out.println("Give col");
        int col = in.nextInt();
        Move playerMove = new Move(row, col, 0);
        board.setLastMove(playerMove, turn % 2);
        board.printBoard();
        turn++;
        System.out.println("Give row");
        row = in.nextInt();
        System.out.println("Give col");
        col = in.nextInt();
        playerMove = new Move(row, col, 0);
        board.setLastMove(playerMove, turn % 2);
        board.printBoard();

    }

}
