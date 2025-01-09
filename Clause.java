import java.util.*;

class Clause {
    private ArrayList<Predicate> premises;
    private Predicate conclusion;

    Clause() {
        this.premises = new ArrayList<>();
        this.conclusion = null;
    }

    Clause(ArrayList<Predicate> premises, Predicate conclusion) {
        this.premises = premises;
        this.conclusion = conclusion;
    }

    public List<Predicate> getPremises() {
        return premises;
    }

    public Predicate getConclusion() {
        return conclusion;
    }

    public void setConclusion(Predicate conc) {
        this.conclusion = conc;
    }

    public boolean isFact() {
        return premises.isEmpty() && conclusion != null;
    }

    public ArrayList<Predicate> getPredicates() {
        ArrayList<Predicate> pred = new ArrayList<>();
        for (Predicate p : this.premises) {
            pred.add(p);
        }
        return pred;
        // pred.add
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj)
            return true;
        if (obj == null || getClass() != obj.getClass())
            return false;
        Clause other = (Clause) obj;
        return Objects.equals(new HashSet<>(premises), new HashSet<>(other.premises)) &&
                Objects.equals(conclusion, other.conclusion);
    }

    @Override
    public int hashCode() {
        return Objects.hash(new HashSet<>(premises), conclusion);
    }

    @Override
    public String toString() {
        return premises + " -> " + conclusion;
    }
}