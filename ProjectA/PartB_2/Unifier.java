import java.util.*;

class Unifier {
    Map<String, String> substitutions = new HashMap<>();

    /**
     * Unifies two Terms and updates the substitution map.
     */
    public boolean unify(Term t1, Term t2) {
        System.out.println("  Unifying terms: " + t1 + " and " + t2);

        // Case 1: Both terms are constants
        if (!t1.isVariable() && !t2.isVariable()) {
            boolean result = t1.getName().equals(t2.getName());
            System.out.println("    Result (constant comparison): " + result);
            return result;
        }

        // Case 2: t1 is a variable
        if (t1.isVariable()) {
            System.out.println("    Substituting " + t1.getName() + " with " + t2.getName());
            return substitute(t1.getName(), t2.getName());
        }

        // Case 3: t2 is a variable
        if (t2.isVariable()) {
            System.out.println("    Substituting " + t2.getName() + " with " + t1.getName());
            return substitute(t2.getName(), t1.getName());
        }

        // Unification failed
        System.out.println("    Unification failed for terms: " + t1 + " and " + t2);
        return false;
    }

    
    public boolean unify(Predicate p1, Predicate p2) {
        
        if (p1.getMembers().size() != p2.getMembers().size()) {
            return false; // Different number of arguments, can't unify
        }
        if (!p1.getName().equals(p2.getName())) {
            return false;
        }

        // Unify each term in the predicate
        for (int i = 0; i < p1.getMembers().size(); i++) {
            Term t1 = p1.getMembers().get(i);
            Term t2 = p2.getMembers().get(i);

            
            if (t1.isVariable) {
                
                if (substitutions.containsKey(t1.name)) {
                    
                    String currentSubstitution = substitutions.get(t1.name);
                    if (!currentSubstitution.equals(t2.getName())) {
                        // Conflict, can't unify
                        return false;
                    }
                } else {
                    
                    if (substitutions.containsValue(t2.getName()))
                        return false;
                    substitutions.put(t1.name, t2.getName());
                }
            } else if (t2.isVariable) {
                
                if (substitutions.containsKey(t2.name)) {
                    String currentSubstitution = substitutions.get(t2.name);
                    if (!currentSubstitution.equals(t1.getName())) {
                        return false; // Conflict
                    }
                } else {
                    if (substitutions.containsValue(t1.getName()))
                        return false;
                    substitutions.put(t2.name, t1.getName());
                }
            } else if (!t1.getName().equals(t2.getName())) {
                
                return false;
            }
        }
        return true;
    }

    
    public Predicate applySubstitution(Predicate p) {
        
        ArrayList<Term> newTerms = new ArrayList<>();

        
        for (Term t : p.getMembers()) {
            if (t.isVariable && substitutions.containsKey(t.name)) {
               
                String substitutedValue = substitutions.get(t.name);
                
                newTerms.add(new Term(substitutedValue, false)); 
            } else {
                
                newTerms.add(t);
            }
        }

        
        return new Predicate(p.getName(), newTerms, p.isNegated);
    }

  
    public Map<String, String> getSubstitutions() {
        return substitutions;
    }

  
    public Term applySubstitution(Term term) {
        if (!term.isVariable()) {
            return term; // Constants are not substituted
        }

        
        String current = term.getName();
        while (substitutions.containsKey(current)) {
            current = substitutions.get(current);
            
            if (current.equals(term.getName()))
                break;
        }

       
        return new Term(current, false);
    }

    
    private boolean substitute(String var, String value) {
       
        if (substitutions.containsKey(var)) {
            return substitutions.get(var).equals(value);
        }

        // Otherwise, add the new substitution
        substitutions.put(var, value);
        return true;
    }

    @Override
    public String toString() {
        return substitutions.toString();
    }
}