import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

public class Loader {

    protected static ArrayList<Clause> clauses;

    public void initialize(ArrayList<Predicate> facts, ArrayList<Clause> clauses) {
        BufferedReader br = null;
        try {
            br = new BufferedReader(new FileReader("KB_fol.txt"));
            String line;
            while (((line = br.readLine()) != null)) {
                if (line.startsWith("FACT:")) {
                    ArrayList<Term> terms = new ArrayList<>();
                    if (line.substring(6).contains("NOT")) {
                        String name = line.substring(10, line.indexOf("(")).trim();
                        String[] parts = line.substring(line.indexOf("(") + 1, line.indexOf(")")).split(",");
                        for (int i = 0; i < parts.length; i++) {
                            parts[i] = parts[i].trim();
                            terms.add(new Term(parts[i], false));
                        }
                        facts.add(new Predicate(name, terms, true));
                    } else {
                        String name = line.substring(6, line.indexOf("(")).trim();
                        String[] parts = line.substring(line.indexOf("(") + 1, line.indexOf(")")).split(",");
                        for (int i = 0; i < parts.length; i++) {
                            parts[i] = parts[i].trim();
                            terms.add(new Term(parts[i], false));
                        }
                        facts.add(new Predicate(name, terms, false));
                    }
                } else if (line.startsWith("CLAUSE: ")) {
                    Clause clause = new Clause();
                    Predicate interLit = null;
                    // Remove the "CLAUSE: " prefix
                    line = line.substring(7).trim();

                    // Split the clause into premises and conclusion
                    String[] tempLine = line.split("->");

                    // Handle premises
                    if (tempLine.length > 0) {
                        String premisesPart = tempLine[0].trim();

                        // Check if there are multiple premises separated by "AND"
                        String[] predicates = premisesPart.contains("AND")
                                ? premisesPart.split("AND")
                                : new String[] { premisesPart };

                        // Process each premise
                        for (String predicateString : predicates) {
                            String[] currPred = predicateString.trim().split("\\(");
                            String name = currPred[0].trim();

                            // Check if the predicate is negated (NOT keyword)
                            if (name.contains("NOT")) {
                                name = name.replace("NOT", "").trim();
                                interLit = new Predicate(name, true); // Negated predicate
                            } else {
                                interLit = new Predicate(name, false); // Positive predicate
                            }

                            // Extract terms inside parentheses
                            if (currPred.length > 1) {
                                String[] terms = currPred[1].substring(0, currPred[1].length() - 1).split(",");
                                for (String term : terms) {
                                    interLit.getMembers().add(new Term(term.trim(), true));
                                }
                            }
                            clause.getPremises().add(interLit);
                        }
                    }

                    // Handle conclusion
                    if (tempLine.length > 1) {
                        Predicate conclusion = null;
                        String[] conclusionParts = tempLine[1].trim().split("\\(");
                        String name = conclusionParts[0].trim();

                        // Check if the conclusion is negated (NOT keyword)
                        if (name.contains("NOT")) {
                            name = name.replace("NOT", "").trim();
                            conclusion = new Predicate(name, true); // Negated conclusion
                        } else {
                            conclusion = new Predicate(name, false); // Positive conclusion
                        }

                        // Extract terms inside parentheses
                        if (conclusionParts.length > 1) {
                            String[] terms = conclusionParts[1].substring(0, conclusionParts[1].length() - 1)
                                    .split(",");
                            for (String term : terms) {
                                conclusion.getMembers().add(new Term(term.trim(), true));
                            }
                        }
                        clause.setConclusion(conclusion);
                    }

                    clauses.add(clause); // Add the parsed clause to the list
                }
            }
            br.close();

        } catch (IOException e) {
            System.out.println("File not found");
        }

    }

    public Predicate getGoal() {
        BufferedReader br = null;
        Predicate goal = null;
        try {
            br = new BufferedReader(new FileReader("KB_fol.txt"));
            String line;
            while (((line = br.readLine()) != null)) {
                if (line.startsWith("PROVE:")) {
                    ArrayList<Term> terms = new ArrayList<>();
                    if (line.substring(7).contains("NOT")) {
                        return goal;
                    } else {
                        String name = line.substring(7, line.indexOf("(")).trim();
                        String[] parts = line.substring(line.indexOf("(") + 1, line.indexOf(")")).split(",");
                        for (int i = 0; i < parts.length; i++) {
                            parts[i] = parts[i].trim();
                            terms.add(new Term(parts[i], false));
                        }
                        goal = new Predicate(name, terms, false);

                    }
                }

            }
            br.close();

        } catch (IOException e) {
            System.out.println("File not found");
        }
        return goal;
    }
}
