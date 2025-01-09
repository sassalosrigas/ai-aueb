import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;

public class Resolver {

    public static boolean resolve(ArrayList<Predicate> facts, ArrayList<Clause> clauses, Predicate goal_literal) {
        Variable_assigner v_m = new Variable_assigner();
        while (true) {
            ArrayList<Clause> new_clauses = new ArrayList<>();
            HashSet<Predicate> new_f = new HashSet<>();
            for (Clause clause : clauses) {
                Clause new_clause = v_m.new_vars(clause);
                Unifier unifier = new Unifier();
                for (Predicate predicate : clause.getPredicates()) {
                    for (Predicate p_fact : facts) {
                        if (unifier.unify(predicate, p_fact)) {
                            ArrayList<Predicate> temp = new ArrayList<>();
                            Predicate beta = unifier.applySubstitution(clause.getConclusion());
                            for (Clause c : clauses) {
                                for (Predicate p : c.getPredicates()) {
                                    temp.add(unifier.applySubstitution(p));
                                }

                            }
                            new_clauses.add(new Clause(temp, beta));
                            if (beta.isFullyInstantiated()) {
                                if (!facts.contains(beta) && !new_f.contains(beta)) {
                                    System.out.println("Adding " + beta.toString());
                                    new_f.add(beta);

                                    System.out.println("Adding conclusion " + unifier.toString());
                                    if (unifier.unify(beta, goal_literal)) {
                                        System.out.println(goal_literal.toString());
                                        System.out.println(beta.toString());
                                        System.out.println(unifier.toString());
                                        return true;

                                    }

                                }

                            }

                        }
                    }

                }
            }

            if (new_f.isEmpty()) {
                break;
            }

            clauses.addAll(new_clauses);
            facts.addAll(new_f);

        }
        return false;

    }

}
