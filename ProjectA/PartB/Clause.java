import java.util.ArrayList;

class Clause {
    private ArrayList<Literal> premises;
    private Literal head;

    public Clause(ArrayList<Literal> prem, Literal h) {
        this.premises = prem;
        this.head = h;
    }

    public ArrayList<Literal> getPremises() {
        return this.premises;
    }

    public Literal getHead() {
        return this.head;
    }

}
