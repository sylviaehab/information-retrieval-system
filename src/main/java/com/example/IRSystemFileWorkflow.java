package com.example;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Information Retrieval System - File-based workflow version
 * Demonstrates the exact Part 1 -> Part 2 workflow as specified in requirements
 */
public class IRSystemFileWorkflow {
    private static final String DATASET_PATH = "dataset";
    private static final String OUTPUT_FILE = "positional_index_output.txt";
    
    public static void main(String[] args) {
        try {
            System.out.println("=".repeat(80));
            System.out.println("INFORMATION RETRIEVAL SYSTEM - FILE-BASED WORKFLOW");
            System.out.println("=".repeat(80));
            
            // PART 1: Build Positional Index and save to output file
            runPart1();
            
            // PART 2: Use the output file from Part 1 for analysis
            runPart2();
            
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * First part (Spark App): Build positional index from Dataset & display each term
     * Save output to file for Part 2
     */
    private static void runPart1() throws IOException {
        System.out.println("\nFIRST PART (SPARK APP): POSITIONAL INDEX CONSTRUCTION");
        System.out.println("-".repeat(60));
        
        // Build positional index from dataset
        Map<String, Map<String, List<Integer>>> positionalIndex = buildPositionalIndex();
        
        // Display positional index
        displayPositionalIndex(positionalIndex);
        
        // Save to output file for Part 2
        savePositionalIndexToFile(positionalIndex, OUTPUT_FILE);
        
        System.out.println("\n✅ Part 1 completed. Output saved to: " + OUTPUT_FILE);
        System.out.println("File contains positional index in required format for Part 2.");
    }
    
    /**
     * Second part: Use the Spark App output file from the First part
     */
    private static void runPart2() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("SECOND PART: USE THE SPARK APP OUTPUT FILE FROM THE FIRST PART");
        System.out.println("=".repeat(80));
        
        // Load positional index from Part 1 output file
        Map<String, Map<String, List<Integer>>> positionalIndex = loadPositionalIndexFromFile(OUTPUT_FILE);
        
        System.out.println("✅ Successfully loaded positional index from: " + OUTPUT_FILE);
        System.out.println("Debug: Loaded " + positionalIndex.size() + " terms");
        if (positionalIndex.size() > 0) {
            System.out.println("Debug: First term example: " + positionalIndex.keySet().iterator().next());
        }
        
        // Get all documents and terms
        List<String> allDocs = getDocumentList();
        Set<String> allTerms = getAllTerms(positionalIndex);
        
        // 2.1 Compute term frequency for each term in each document (Display it)
        System.out.println("\n2.1 TERM FREQUENCY MATRIX");
        System.out.println("-".repeat(40));
        Map<String, Map<String, Integer>> tfMatrix = computeTermFrequency(positionalIndex, allDocs, allTerms);
        displayTermFrequencyMatrix(tfMatrix, allDocs, allTerms);
        
        // 2.2 Compute IDF for each term (Display it)
        System.out.println("\n2.2 IDF VALUES");
        System.out.println("-".repeat(40));
        Map<String, Double> idfValues = computeIDF(positionalIndex, allDocs.size());
        displayIDFValues(idfValues);
        
        // 2.3 Compute TF.IDF matrix for each term (Display it)
        System.out.println("\n2.3 TF-IDF MATRIX");
        System.out.println("-".repeat(40));
        Map<String, Map<String, Double>> tfidfMatrix = computeTFIDF(tfMatrix, idfValues, allDocs, allTerms);
        displayTFIDFMatrix(tfidfMatrix, allDocs, allTerms);
        
        // 2.4 Allow users to enter phrase queries with boolean operators
        System.out.println("\n2.4 PHRASE QUERIES WITH BOOLEAN OPERATORS");
        System.out.println("-".repeat(50));
        processQuery("mercy AND caeser", positionalIndex, tfidfMatrix, allDocs);
        processQuery("brutus AND NOT mercy", positionalIndex, tfidfMatrix, allDocs);
        processQuery("angels AND fools", positionalIndex, tfidfMatrix, allDocs);
    }
    
    // === IMPLEMENTATION METHODS ===
    
    private static Map<String, Map<String, List<Integer>>> buildPositionalIndex() throws IOException {
        File datasetDir = new File(DATASET_PATH);
        File[] files = datasetDir.listFiles((dir, name) -> name.endsWith(".txt"));
        
        if (files == null || files.length == 0) {
            throw new RuntimeException("No dataset files found in " + DATASET_PATH);
        }
        
        Map<String, Map<String, List<Integer>>> positionalIndex = new HashMap<>();
        
        for (File file : files) {
            String docId = file.getName().replace(".txt", "");
            String content = Files.readString(Paths.get(file.getAbsolutePath())).trim().toLowerCase();
            
            String[] terms = content.split("\\s+");
            
            for (int position = 0; position < terms.length; position++) {
                String term = terms[position].trim();
                if (!term.isEmpty()) {
                    positionalIndex.computeIfAbsent(term, k -> new HashMap<>())
                                  .computeIfAbsent(docId, k -> new ArrayList<>())
                                  .add(position + 1); // 1-based positioning
                }
            }
        }
        
        return positionalIndex;
    }
    
    private static void displayPositionalIndex(Map<String, Map<String, List<Integer>>> positionalIndex) {
        System.out.println("Positional Index:");
        
        positionalIndex.entrySet().stream()
                .sorted(Map.Entry.comparingByKey())
                .forEach(entry -> {
                    String term = entry.getKey();
                    System.out.print("<" + term + " ");
                    
                    entry.getValue().entrySet().stream()
                            .sorted((e1, e2) -> Integer.compare(Integer.parseInt(e1.getKey()), Integer.parseInt(e2.getKey())))
                            .forEach(docEntry -> {
                                String docId = docEntry.getKey();
                                List<Integer> positions = docEntry.getValue();
                                System.out.print("doc" + docId + ": " + 
                                        positions.stream()
                                                .map(String::valueOf)
                                                .collect(Collectors.joining(", ")) + " ");
                            });
                    
                    System.out.println(">");
                });
    }
    
    private static void savePositionalIndexToFile(Map<String, Map<String, List<Integer>>> positionalIndex, 
                                                  String fileName) {
        try (PrintWriter writer = new PrintWriter(new FileWriter(fileName))) {
            writer.println("=== SPARK APP OUTPUT FILE - POSITIONAL INDEX ===");
            writer.println("Generated by First part (Spark App) of Information Retrieval System");
            writer.println("Format: <term doc1: position1, position2... doc2: position1, position2... etc.>");
            writer.println();
            
            positionalIndex.entrySet().stream()
                    .sorted(Map.Entry.comparingByKey())
                    .forEach(entry -> {
                        String term = entry.getKey();
                        writer.print("<" + term + " ");
                        
                        entry.getValue().entrySet().stream()
                                .sorted((e1, e2) -> Integer.compare(Integer.parseInt(e1.getKey()), Integer.parseInt(e2.getKey())))
                                .forEach(docEntry -> {
                                    String docId = docEntry.getKey();
                                    List<Integer> positions = docEntry.getValue();
                                    writer.print("doc" + docId + ": " + 
                                            positions.stream()
                                                    .map(String::valueOf)
                                                    .collect(Collectors.joining(", ")) + " ");
                                });
                        
                        writer.println(">");
                    });
                    
            writer.println();
            writer.println("=== END OF POSITIONAL INDEX OUTPUT ===");
            
        } catch (IOException e) {
            throw new RuntimeException("Error writing positional index to file: " + fileName, e);
        }
    }
    
    private static Map<String, Map<String, List<Integer>>> loadPositionalIndexFromFile(String fileName) {
        Map<String, Map<String, List<Integer>>> positionalIndex = new HashMap<>();
        
        try {
            List<String> lines = Files.readAllLines(Paths.get(fileName));
            
            for (String line : lines) {
                line = line.trim();
                if (line.startsWith("<") && line.endsWith(">")) {
                    // Parse line like: <angels doc7: 1 doc8: 1 doc9: 1 >
                    String content = line.substring(1, line.length() - 1).trim();
                    
                    // Use regex to find term and document entries
                    String[] parts = content.split("\\s+");
                    if (parts.length > 0) {
                        String term = parts[0];
                        Map<String, List<Integer>> termDocs = new HashMap<>();
                        
                        // Process remaining parts to find docX: positions
                        for (int i = 1; i < parts.length; i++) {
                            if (parts[i].startsWith("doc") && parts[i].contains(":")) {
                                String docPart = parts[i];
                                String[] docSplit = docPart.split(":");
                                if (docSplit.length == 2) {
                                    String docId = docSplit[0].substring(3); // Remove "doc" prefix
                                    String positionsStr = docSplit[1];
                                    
                                    List<Integer> positions = new ArrayList<>();
                                    String[] posArray = positionsStr.split(",");
                                    for (String pos : posArray) {
                                        try {
                                            String cleanPos = pos.trim().replaceAll("[^0-9]", "");
                                            if (!cleanPos.isEmpty()) {
                                                positions.add(Integer.parseInt(cleanPos));
                                            }
                                        } catch (NumberFormatException e) {
                                            // Skip invalid positions
                                        }
                                    }
                                    
                                    if (!positions.isEmpty()) {
                                        termDocs.put(docId, positions);
                                    }
                                }
                            }
                        }
                        
                        if (!termDocs.isEmpty()) {
                            positionalIndex.put(term, termDocs);
                        }
                    }
                }
            }
            
        } catch (IOException e) {
            throw new RuntimeException("Error reading positional index from file: " + fileName, e);
        }
        
        return positionalIndex;
    }
    
    // === ANALYSIS METHODS (Same as Main.java) ===
    
    private static List<String> getDocumentList() {
        List<String> docs = new ArrayList<>();
        for (int i = 1; i <= 10; i++) {
            docs.add(String.valueOf(i));
        }
        return docs;
    }
    
    private static Set<String> getAllTerms(Map<String, Map<String, List<Integer>>> positionalIndex) {
        return new TreeSet<>(positionalIndex.keySet());
    }
    
    private static Map<String, Map<String, Integer>> computeTermFrequency(
            Map<String, Map<String, List<Integer>>> positionalIndex,
            List<String> allDocs, Set<String> allTerms) {
        
        Map<String, Map<String, Integer>> tfMatrix = new HashMap<>();
        
        for (String term : allTerms) {
            Map<String, Integer> termFreqs = new HashMap<>();
            Map<String, List<Integer>> termDocs = positionalIndex.get(term);
            
            for (String doc : allDocs) {
                if (termDocs != null && termDocs.containsKey(doc)) {
                    termFreqs.put(doc, termDocs.get(doc).size());
                } else {
                    termFreqs.put(doc, 0);
                }
            }
            tfMatrix.put(term, termFreqs);
        }
        
        return tfMatrix;
    }
    
    private static void displayTermFrequencyMatrix(Map<String, Map<String, Integer>> tfMatrix,
                                                   List<String> allDocs, Set<String> allTerms) {
        System.out.println("\nTerm Frequency Matrix:");
        
        System.out.printf("%-12s", "Term");
        for (String doc : allDocs) {
            System.out.printf("%8s", "Doc" + doc);
        }
        System.out.println();
        System.out.println("-".repeat(12 + allDocs.size() * 8));
        
        for (String term : allTerms) {
            System.out.printf("%-12s", term);
            Map<String, Integer> termFreqs = tfMatrix.get(term);
            for (String doc : allDocs) {
                System.out.printf("%8d", termFreqs.get(doc));
            }
            System.out.println();
        }
    }
    
    private static Map<String, Double> computeIDF(Map<String, Map<String, List<Integer>>> positionalIndex,
                                                  int totalDocs) {
        Map<String, Double> idfValues = new HashMap<>();
        
        for (String term : positionalIndex.keySet()) {
            int df = positionalIndex.get(term).size();
            double idf = Math.log10((double) totalDocs / df);
            idfValues.put(term, idf);
        }
        
        return idfValues;
    }
    
    private static void displayIDFValues(Map<String, Double> idfValues) {
        System.out.println("\nIDF Values:");
        System.out.printf("%-12s %8s\n", "Term", "IDF");
        System.out.println("-".repeat(22));
        
        idfValues.entrySet().stream()
                .sorted(Map.Entry.comparingByKey())
                .forEach(entry -> {
                    System.out.printf("%-12s %8.4f\n", entry.getKey(), entry.getValue());
                });
    }
    
    private static Map<String, Map<String, Double>> computeTFIDF(
            Map<String, Map<String, Integer>> tfMatrix,
            Map<String, Double> idfValues,
            List<String> allDocs, Set<String> allTerms) {
        
        Map<String, Map<String, Double>> tfidfMatrix = new HashMap<>();
        
        for (String term : allTerms) {
            Map<String, Double> termTFIDF = new HashMap<>();
            Map<String, Integer> termFreqs = tfMatrix.get(term);
            double idf = idfValues.get(term);
            
            for (String doc : allDocs) {
                int tf = termFreqs.get(doc);
                double tfidf;
                if (tf > 0) {
                    double logTF = 1 + Math.log10(tf); // Using 1+log10(tf) rule
                    tfidf = logTF * idf;
                } else {
                    tfidf = 0.0;
                }
                termTFIDF.put(doc, tfidf);
            }
            tfidfMatrix.put(term, termTFIDF);
        }
        
        return tfidfMatrix;
    }
    
    private static void displayTFIDFMatrix(Map<String, Map<String, Double>> tfidfMatrix,
                                           List<String> allDocs, Set<String> allTerms) {
        System.out.println("\nTF-IDF Matrix (using 1+log10(tf) rule):");
        
        System.out.printf("%-12s", "Term");
        for (String doc : allDocs) {
            System.out.printf("%10s", "Doc" + doc);
        }
        System.out.println();
        System.out.println("-".repeat(12 + allDocs.size() * 10));
        
        for (String term : allTerms) {
            System.out.printf("%-12s", term);
            Map<String, Double> termTFIDF = tfidfMatrix.get(term);
            for (String doc : allDocs) {
                System.out.printf("%10.4f", termTFIDF.get(doc));
            }
            System.out.println();
        }
    }
    
    private static void processQuery(String query, 
                                   Map<String, Map<String, List<Integer>>> positionalIndex,
                                   Map<String, Map<String, Double>> tfidfMatrix,
                                   List<String> allDocs) {
        System.out.println("\nProcessing Query: \"" + query + "\"");
        System.out.println("-".repeat(30));
        
        Set<String> resultDocs = evaluateQuery(query, positionalIndex);
        
        if (resultDocs.isEmpty()) {
            System.out.println("No matching documents found.");
            return;
        }
        
        List<DocumentScore> rankedDocs = calculateSimilarity(query, resultDocs, tfidfMatrix);
        
        System.out.println("Matching Documents (ranked by similarity):");
        for (int i = 0; i < rankedDocs.size(); i++) {
            DocumentScore docScore = rankedDocs.get(i);
            System.out.printf("%d. Doc%s (Score: %.4f)\n", i + 1, docScore.docId, docScore.score);
        }
    }
    
    private static Set<String> evaluateQuery(String query, 
                                           Map<String, Map<String, List<Integer>>> positionalIndex) {
        query = query.toLowerCase().trim();
        
        if (query.contains(" and not ")) {
            String[] parts = query.split(" and not ");
            String positiveTerm = parts[0].trim();
            String negativeTerm = parts[1].trim();
            
            Set<String> positiveResults = getDocsContaining(positiveTerm, positionalIndex);
            Set<String> negativeResults = getDocsContaining(negativeTerm, positionalIndex);
            
            positiveResults.removeAll(negativeResults);
            return positiveResults;
            
        } else if (query.contains(" and ")) {
            String[] terms = query.split(" and ");
            Set<String> result = getDocsContaining(terms[0].trim(), positionalIndex);
            
            for (int i = 1; i < terms.length; i++) {
                Set<String> termDocs = getDocsContaining(terms[i].trim(), positionalIndex);
                result.retainAll(termDocs);
            }
            return result;
        } else {
            return getDocsContaining(query, positionalIndex);
        }
    }
    
    private static Set<String> getDocsContaining(String term, 
                                               Map<String, Map<String, List<Integer>>> positionalIndex) {
        if (positionalIndex.containsKey(term)) {
            return new HashSet<>(positionalIndex.get(term).keySet());
        }
        return new HashSet<>();
    }
    
    private static List<DocumentScore> calculateSimilarity(String query,
                                                           Set<String> candidateDocs,
                                                           Map<String, Map<String, Double>> tfidfMatrix) {
        String[] queryTerms = query.toLowerCase()
                .replaceAll(" and not | and ", " ")
                .split("\\s+");
        
        List<DocumentScore> scores = new ArrayList<>();
        
        for (String doc : candidateDocs) {
            double score = 0.0;
            for (String term : queryTerms) {
                if (tfidfMatrix.containsKey(term)) {
                    score += tfidfMatrix.get(term).get(doc);
                }
            }
            scores.add(new DocumentScore(doc, score));
        }
        
        scores.sort((a, b) -> Double.compare(b.score, a.score));
        
        return scores;
    }
    
    static class DocumentScore {
        String docId;
        double score;
        
        public DocumentScore(String docId, double score) {
            this.docId = docId;
            this.score = score;
        }
    }
}