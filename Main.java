import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;;

public class Main {

    public static void main(String Args[]) {
        ArrayList<Predicate> facts = new ArrayList<>();
        ArrayList<Clause> clauses = new ArrayList<>();
        Loader l = new Loader();
        l.initialize(facts, clauses);
        Predicate goal_predicate = l.getGoal();
        System.out.println(goal_predicate);
        Scanner in = new Scanner(System.in);
        boolean can_be_proven = Resolver.resolve(facts, clauses, goal_predicate);
        if (can_be_proven) {
            System.out.println("The given conclusion can be proven");
        } else {
            System.out.println("The given conclusion can't be proven with the current information");
        }
        in.close();
    }

}
