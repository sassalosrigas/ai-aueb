public class Literal {
    private String name;
    private boolean isNegated;

    Literal(String name, boolean neg) {
        this.name = name;
        this.isNegated = neg;
    }

    boolean isNegated() {
        return this.isNegated;
    }

    String getName() {
        return this.name;
    }

    void print() {
        if (this.isNegated)
            System.out.println("NOT " + this.name);
        else
            System.out.println(this.name);
    }

    @Override
    public boolean equals(Object obj) {
        Literal l = (Literal) obj;
        if (l.getName().equals(this.name) && l.isNegated() == this.isNegated)
            return true;
        else
            return false;
    }

    @Override
    public int hashCode() {
        if (this.isNegated)
            return this.name.hashCode() + 1;
        else
            return this.name.hashCode();
    }
}