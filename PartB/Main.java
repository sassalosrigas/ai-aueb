import java.io.*;
import java.util.*;

class Main {

    public static void main(String Args[]) {
        try {
            BufferedReader br = new BufferedReader(new FileReader("horn_clause_KB.txt"));
            String line;
            ArrayList<Literal> literals = new ArrayList<Literal>();
            Hashtable<Clause, Integer> count = new Hashtable<>();
            ArrayList<Clause> clauses = new ArrayList<>();
            HashMap<String, Boolean> inferred = new HashMap<>();
            Queue<Literal> agenda = new LinkedList<>();
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.startsWith("FACT:")) { // Eksagogh literals
                    String[] fact = line.substring(6).split(",");
                    fact[0] = fact[0].trim();
                    Literal literal;
                    if (fact[0].substring(0, 1).equals("!")) {
                        literal = new Literal(fact[0].substring(1).trim(), true);
                    } else {
                        literal = new Literal(fact[0].trim(), false);
                    }
                    literals.add(literal);
                    agenda.add(literal);
                    inferred.put(literal.getName(), false);
                } else if (line.startsWith("CLAUSE:")) {
                    String[] parts = line.split("->");
                    String[] prems = parts[0].substring(8).split("AND");
                    String head = parts[1].trim();
                    Literal head_l;
                    if (head.substring(0, 1).equals("!")) {
                        head_l = new Literal(head.substring(1).trim(), true);
                    } else {
                        head_l = new Literal(head.trim(), false);
                    }
                    ArrayList<Literal> premises = new ArrayList<>();
                    for (String p : prems) {
                        if (p.substring(0, 1).equals("!")) {
                            premises.add(new Literal(p.substring(1).trim(), true));
                        } else {
                            premises.add(new Literal(p.trim(), false));
                        }
                        inferred.put(p.trim(), false);
                    }
                    Clause clause = new Clause(premises, head_l);
                    clauses.add(clause);
                    count.put(clause, premises.size());
                    inferred.put(head_l.getName(), false);

                }
            }
            br.close();

            Literal goalLiteral = new Literal("Q", false);
            boolean sat = resolve_fc(agenda, count, clauses, inferred, goalLiteral);
            if (sat) {
                System.out.println("Satisfiable");
            } else {
                System.out.println("Unsatisfiable");
            }
        } catch (IOException e) {
            System.out.println("File not found");
        }

    }

    public static boolean resolve_fc(Queue<Literal> agenda, Hashtable<Clause, Integer> count, ArrayList<Clause> clauses,
            HashMap<String, Boolean> inferred, Literal goal_l) {
        while (!agenda.isEmpty()) {
            Literal l = agenda.poll();
            if (!inferred.get(l.getName())) {
                inferred.replace(l.getName(), true);
                for (Clause clause : clauses) {
                    if (clause.getPremises().contains(l)) {
                        count.replace(clause, count.get(clause) - 1);
                    }
                    if (count.get(clause) == 0) {
                        Literal head = clause.getHead();
                        if (head.equals(goal_l)) {
                            return true;
                        }
                        if (!inferred.get(head.getName())) {
                            agenda.add(head);
                        }
                    }
                }
            }
        }
        return false;
    }
}
