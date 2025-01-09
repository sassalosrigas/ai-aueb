import java.util.*;

class Predicate {
    String name;
    List<Term> terms;
    boolean isNegated;

    Predicate(String name, boolean isNegated) {
        this.name = name;
        this.terms = new ArrayList<Term>();
        this.isNegated = isNegated;
    }

    Predicate(String name, List<Term> terms, boolean isNegated) {
        this.name = name;
        this.terms = terms;
        this.isNegated = isNegated;
    }

    public List<Term> getMembers() {
        return this.terms;
    }

    public String getName() {
        return this.name;
    }

    public void negate() {
        this.isNegated = !this.isNegated;
    }

    public boolean isCopy(Predicate pred) {
        if (this.name != pred.getName() || this.terms.size() != pred.terms.size()) {
            return false;
        }
        for (int i = 0; i < this.terms.size(); i++) {
            if ((this.terms.get(i).isVariable() && !pred.terms.get(i).isVariable())
                    || (!this.terms.get(i).isVariable() && pred.terms.get(i).isVariable())) {
                return false;
            }
        }
        return true;
    }

    public boolean isFullyInstantiated() {
        for (Term term : this.getMembers()) {
            if (term.isVariable()) {
                return false; // Contains an unresolved variable
            }
        }
        return true; // All terms are constants
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj)
            return true;
        if (obj == null || getClass() != obj.getClass())
            return false;
        Predicate other = (Predicate) obj;
        return isNegated == other.isNegated && name.equals(other.name) && terms.equals(other.terms);
    }

    @Override
    public int hashCode() {
        return Objects.hash(name, terms, isNegated);
    }

    @Override
    public String toString() {
        return (isNegated ? "~" : "") + name + "(" + terms + ")";
    }

    boolean isNegation(Predicate other) {
        return this.name.equals(other.name) && this.isNegated != other.isNegated;
    }
}
