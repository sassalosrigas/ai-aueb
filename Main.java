import java.util.Scanner;

class Main{

    public static void main(String args[]){
        Scanner in = new Scanner(System.in);
        System.out.println("Give depth");
        int depth = in.nextInt();
        System.out.println("Do you want to play first? Y(Yes) N(No)");
        String first = in.nextLine();
        first = in.nextLine();
        Board board = new Board(8);
        board.initializeBoard();
        board.printBoard();
        
    }

}
