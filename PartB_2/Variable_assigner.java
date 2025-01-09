import java.util.ArrayList;
import java.util.HashMap;

public class Variable_assigner {
    private static int var_count = 0;
    private static HashMap<String, String> variableMap = new HashMap<>(); // Tracks renaming per clause

    public Clause new_vars(Clause clause) {
        var_count = 0;
        ArrayList<Predicate> preds = new ArrayList<>();

        // Process the premises
        for (Predicate premise : clause.getPremises()) {
            ArrayList<Term> terms = new ArrayList<>();
            for (Term term : premise.getMembers()) {
                if (term.isVariable) {
                    // Ensure consistent renaming for each variable within the clause
                    if (!variableMap.containsKey(term.name)) {
                        variableMap.put(term.name, "v" + var_count++);
                    }
                    terms.add(new Term(variableMap.get(term.name), true));
                } else {
                    terms.add(term);
                }
            }
            preds.add(new Predicate(premise.getName(), terms, premise.isNegated));
        }

        // Process the conclusion (if it exists)
        Predicate conclusion = clause.getConclusion();
        if (conclusion != null) {
            ArrayList<Term> terms = new ArrayList<>();
            for (Term term : conclusion.getMembers()) {
                if (term.isVariable) {
                    // Ensure consistent renaming for each variable within the clause
                    if (!variableMap.containsKey(term.name)) {
                        variableMap.put(term.name, "v" + var_count++);
                    }
                    terms.add(new Term(variableMap.get(term.name), true));
                } else {
                    terms.add(term);
                }
            }
            conclusion = new Predicate(conclusion.getName(), terms,
                    conclusion.isNegated);
        }

        return new Clause(preds, conclusion);
    }

}
