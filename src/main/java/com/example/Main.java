package com.example;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;

public class Main {
    private static final String DATASET_PATH = "dataset";
    
    public static void main(String[] args) {
        try {
            System.out.println("=".repeat(80));
            System.out.println("INFORMATION RETRIEVAL SYSTEM");
            System.out.println("=".repeat(80));
            
            // Part 1: Build Positional Index
            System.out.println("\nPART 1: POSITIONAL INDEX");
            System.out.println("-".repeat(50));
            Map<String, Map<String, List<Integer>>> positionalIndex = buildPositionalIndex();
            displayPositionalIndex(positionalIndex);
            
            // Part 2: Compute Term Frequencies, IDF, and TF-IDF
            System.out.println("\nPART 2: TERM FREQUENCY ANALYSIS");
            System.out.println("-".repeat(50));
            
            // Get all documents and terms
            List<String> allDocs = getDocumentList();
            Set<String> allTerms = getAllTerms(positionalIndex);
            
            // Compute and display Term Frequency Matrix
            Map<String, Map<String, Integer>> tfMatrix = computeTermFrequency(positionalIndex, allDocs, allTerms);
            displayTermFrequencyMatrix(tfMatrix, allDocs, allTerms);
            
            // Compute and display IDF values
            Map<String, Double> idfValues = computeIDF(positionalIndex, allDocs.size());
            displayIDFValues(idfValues);
            
            // Compute and display TF-IDF Matrix using 1+log10(tf) rule
            Map<String, Map<String, Double>> tfidfMatrix = computeTFIDF(tfMatrix, idfValues, allDocs, allTerms);
            displayTFIDFMatrix(tfidfMatrix, allDocs, allTerms);
            
            // Part 3: Query Processing
            System.out.println("\nPART 3: QUERY PROCESSING");
            System.out.println("-".repeat(50));
            
            // Demo queries
            processQuery("mercy AND caeser", positionalIndex, tfidfMatrix, allDocs);
            processQuery("brutus AND NOT mercy", positionalIndex, tfidfMatrix, allDocs);
            processQuery("angels AND fools", positionalIndex, tfidfMatrix, allDocs);
            
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private static Map<String, Map<String, List<Integer>>> buildPositionalIndex() throws IOException {
        // Read all document files
        File datasetDir = new File(DATASET_PATH);
        File[] files = datasetDir.listFiles((dir, name) -> name.endsWith(".txt"));
        
        if (files == null || files.length == 0) {
            throw new RuntimeException("No dataset files found in " + DATASET_PATH);
        }
        
        Map<String, Map<String, List<Integer>>> positionalIndex = new HashMap<>();
        
        for (File file : files) {
            String docId = file.getName().replace(".txt", "");
            
            // Read document content
            String content = Files.readString(Paths.get(file.getAbsolutePath())).trim();
            
            // Tokenize and build positional index
            String[] terms = content.toLowerCase().split("\\s+");
            
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
        
        // Header
        System.out.printf("%-12s", "Term");
        for (String doc : allDocs) {
            System.out.printf("%8s", "Doc" + doc);
        }
        System.out.println();
        System.out.println("-".repeat(12 + allDocs.size() * 8));
        
        // Data rows
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
            int df = positionalIndex.get(term).size(); // Document frequency
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
                    // Using the rule: TF = 1 + log10(tf)
                    double logTF = 1 + Math.log10(tf);
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
        
        // Header
        System.out.printf("%-12s", "Term");
        for (String doc : allDocs) {
            System.out.printf("%10s", "Doc" + doc);
        }
        System.out.println();
        System.out.println("-".repeat(12 + allDocs.size() * 10));
        
        // Data rows
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
        
        // Calculate similarity scores for matching documents
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
        
        // Sort by score in descending order
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