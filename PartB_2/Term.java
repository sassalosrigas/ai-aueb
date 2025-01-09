import java.util.Objects;

class Term {
    String name;
    boolean isVariable;

    Term(String name, boolean isVariable) {
        this.name = name;
        this.isVariable = isVariable;
    }

    Term(String name) {
        this.name = name;
        if (name.length() == 1) {
            // The string is made up of one character...
            this.isVariable = true;
        }
    }

    public String getName() {
        return this.name;
    }

    public boolean isVariable() {
        return this.isVariable;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) {
            return true; // Same object
        }
        if (obj == null || getClass() != obj.getClass()) {
            return false; // Different class
        }
        Term other = (Term) obj;
        return this.name.equals(other.name) &&
                this.isVariable == other.isVariable;
    }

    @Override
    public int hashCode() {
        return Objects.hash(name, isVariable);
    }

    @Override
    public String toString() {
        return name;
    }
}