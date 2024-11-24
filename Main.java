import java.util.Scanner;

class Main {

    public static void main(String args[]) {
        int ai = 0;
        Scanner in = new Scanner(System.in);
        System.out.print("Give depth: ");
        int depth = in.nextInt();
        in.nextLine();
        boolean f = false;
        while (f == false) { // Epilogh prwtou paixth
            System.out.print("Do you want to play first? Y(Yes) N(No): ");
            String first = in.nextLine();
            if (first.equals("Y")) {
                f = true;
                ai = 1;
                System.out.println("You are black");
            } else if (first.equals("N")) {
                f = true;
                ai = -1;
                System.out.println("You are white");
            } else {
                System.out.println("Invalid input");
            }
        }
        Player playerBlack = new Player(depth, Board.B);
        Player playerWhite = new Player(depth, Board.W);
        Board board = new Board(); // arxikopoihsh kai ektypwsh pinaka
        board.printBoard();
        if (ai == -1) { // Ai prwto xrhsths deuteros epilogh "N"
            while (!board.isTerminal()) {
                switch (board.getLastPlayer()) {
                    case Board.W:
                        System.out.println("Black plays");
                        Move moveB = playerBlack.MiniMax(board); // Klhsh minimax gia automath epilogh kinhshs
                        board.makeMove(moveB.getRow(), moveB.getCol(), Board.B);
                        break;
                    case Board.B:
                        System.out.println("White plays");

                        System.out.print("Give row: "); // Diavasma suntetagmenwn apo xrhsth
                        int row = in.nextInt();
                        System.out.print("Give col: ");
                        int col = in.nextInt();
                        row -= 1; // Diorthosi sintetagmenwn gia na tairiazoun sta boundaries tou array tou board
                        col -= 1;
                        Move moveW = new Move(row, col);
                        board.makeMove(moveW.getRow(), moveW.getCol(), Board.W);
                        break;
                    default:
                        break;
                }
                board.printBoard(); // Ektiposh pinaka meta apo kathe kinhsh
            }
        } else if (ai == 1) { // Xrhsths prwtos ai deutero epilogh "Y"
            while (!board.isTerminal()) {
                switch (board.getLastPlayer()) {
                    case Board.W:
                        System.out.println("Black plays");

                        System.out.print("Give row: ");
                        int row = in.nextInt();
                        System.out.print("Give col: ");
                        int col = in.nextInt();
                        row -= 1;
                        col -= 1;
                        Move moveB = new Move(row, col, board.evaluate());
                        board.makeMove(moveB.getRow(), moveB.getCol(), Board.B);
                        break;
                    case Board.B:
                        System.out.println("White plays");
                        Move moveW = playerWhite.MiniMax(board);
                        board.makeMove(moveW.getRow(), moveW.getCol(), Board.W);
                        break;
                    default:
                        break;
                }
                board.printBoard();
            }
        }
        in.close();
    }
}
