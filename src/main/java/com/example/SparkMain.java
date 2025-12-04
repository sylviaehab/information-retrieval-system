package com.example;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;

import java.io.File;
import java.io.Serializable;
import java.util.*;
import java.util.stream.Collectors;

public class SparkMain implements Serializable {
    private static final String DATASET_PATH = "dataset";
    
    public static void main(String[] args) {
        // Configure Spark
        SparkConf conf = new SparkConf()
                .setAppName("Information Retrieval System with Spark")
                .setMaster("local[*]")
                .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
                .set("spark.sql.adaptive.enabled", "true");
        
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("WARN");
        
        try {
            System.out.println("=".repeat(80));
            System.out.println("INFORMATION RETRIEVAL SYSTEM");
            System.out.println("=".repeat(80));
            
            // Part 1: Build Positional Index using Spark
            System.out.println("\nPART 1: POSITIONAL INDEX");
            System.out.println("-".repeat(50));
            JavaPairRDD<String, Map<String, List<Integer>>> positionalIndexRDD = buildPositionalIndexWithSpark(sc);
            Map<String, Map<String, List<Integer>>> positionalIndex = positionalIndexRDD.collectAsMap();
            displayPositionalIndex(positionalIndex);
            
            // Part 2: Compute Term Frequencies, IDF, and TF-IDF using Spark
            System.out.println("\nPART 2: TERM FREQUENCY ANALYSIS");
            System.out.println("-".repeat(50));
            
            // Get all documents and terms
            List<String> allDocs = getDocumentList();
            Set<String> allTerms = getAllTerms(positionalIndex);
            
            // Compute TF-IDF using Spark RDDs
            JavaPairRDD<String, Map<String, Double>> tfidfRDD = computeTFIDFWithSpark(sc, positionalIndex, allDocs, allTerms);
            Map<String, Map<String, Double>> tfidfMatrix = tfidfRDD.collectAsMap();
            
            // Display results
            Map<String, Map<String, Integer>> tfMatrix = computeTermFrequency(positionalIndex, allDocs, allTerms);
            displayTermFrequencyMatrix(tfMatrix, allDocs, allTerms);
            
            Map<String, Double> idfValues = computeIDF(positionalIndex, allDocs.size());
            displayIDFValues(idfValues);
            
            displayTFIDFMatrix(tfidfMatrix, allDocs, allTerms);
            
            // Part 3: Query Processing with Spark
            System.out.println("\nPART 3: QUERY PROCESSING");
            System.out.println("-".repeat(50));
            
            processQueryWithSpark(sc, "mercy AND caeser", positionalIndex, tfidfMatrix, allDocs);
            processQueryWithSpark(sc, "brutus AND NOT mercy", positionalIndex, tfidfMatrix, allDocs);
            processQueryWithSpark(sc, "angels AND fools", positionalIndex, tfidfMatrix, allDocs);
            
        } finally {
            sc.close();
        }
    }
    
    private static JavaPairRDD<String, Map<String, List<Integer>>> buildPositionalIndexWithSpark(JavaSparkContext sc) {
        // Read all document files and create RDD
        File datasetDir = new File(DATASET_PATH);
        File[] files = datasetDir.listFiles((dir, name) -> name.endsWith(".txt"));
        
        if (files == null || files.length == 0) {
            throw new RuntimeException("No dataset files found in " + DATASET_PATH);
        }
        
        // Create list of file paths
        List<String> filePaths = Arrays.stream(files)
                .map(File::getAbsolutePath)
                .collect(Collectors.toList());
        
        // Create RDD from file paths
        JavaRDD<String> filePathsRDD = sc.parallelize(filePaths);
        
        // Process each file and create positional index entries
        JavaPairRDD<String, Tuple2<String, List<Integer>>> termDocPositionsRDD = filePathsRDD.flatMapToPair(filePath -> {
            List<Tuple2<String, Tuple2<String, List<Integer>>>> termEntries = new ArrayList<>();
            
            try {
                String docId = new File(filePath).getName().replace(".txt", "");
                
                // Read file content directly without creating nested RDD
                java.nio.file.Path path = java.nio.file.Paths.get(filePath);
                String content = java.nio.file.Files.readString(path).toLowerCase().trim();
                
                String[] terms = content.split("\\s+");
                
                Map<String, List<Integer>> termPositions = new HashMap<>();
                for (int position = 0; position < terms.length; position++) {
                    String term = terms[position].trim();
                    if (!term.isEmpty()) {
                        termPositions.computeIfAbsent(term, k -> new ArrayList<>())
                                .add(position + 1);
                    }
                }
                
                for (Map.Entry<String, List<Integer>> entry : termPositions.entrySet()) {
                    termEntries.add(new Tuple2<>(entry.getKey(), new Tuple2<>(docId, entry.getValue())));
                }
            } catch (Exception e) {
                throw new RuntimeException("Error processing file: " + filePath, e);
            }
            
            return termEntries.iterator();
        });
        
        // Group by term and collect all document positions
        JavaPairRDD<String, Map<String, List<Integer>>> positionalIndexRDD = termDocPositionsRDD
                .groupByKey()
                .mapValues(docPositions -> {
                    Map<String, List<Integer>> docMap = new HashMap<>();
                    for (Tuple2<String, List<Integer>> docPos : docPositions) {
                        docMap.put(docPos._1(), docPos._2());
                    }
                    return docMap;
                });
        
        return positionalIndexRDD;
    }
    
    private static JavaPairRDD<String, Map<String, Double>> computeTFIDFWithSpark(
            JavaSparkContext sc,
            Map<String, Map<String, List<Integer>>> positionalIndex,
            List<String> allDocs,
            Set<String> allTerms) {
        
        // Create RDD from terms
        JavaRDD<String> termsRDD = sc.parallelize(new ArrayList<>(allTerms));
        
        int totalDocs = allDocs.size();
        
        // Compute TF-IDF for each term
        JavaPairRDD<String, Map<String, Double>> tfidfRDD = termsRDD.mapToPair(term -> {
            Map<String, List<Integer>> termDocs = positionalIndex.get(term);
            int df = termDocs.size(); // Document frequency
            double idf = Math.log10((double) totalDocs / df);
            
            Map<String, Double> termTFIDF = new HashMap<>();
            
            for (String doc : allDocs) {
                double tfidf;
                if (termDocs.containsKey(doc)) {
                    int tf = termDocs.get(doc).size();
                    double logTF = 1 + Math.log10(tf); // Using 1+log10(tf) rule
                    tfidf = logTF * idf;
                } else {
                    tfidf = 0.0;
                }
                termTFIDF.put(doc, tfidf);
            }
            
            return new Tuple2<>(term, termTFIDF);
        });
        
        return tfidfRDD;
    }
    
    private static void processQueryWithSpark(JavaSparkContext sc,
                                            String query,
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
        
        // Use Spark to calculate similarity scores
        JavaRDD<String> candidateDocsRDD = sc.parallelize(new ArrayList<>(resultDocs));
        
        String[] queryTerms = query.toLowerCase()
                .replaceAll(" and not | and ", " ")
                .split("\\s+");
        
        JavaPairRDD<String, Double> scoresRDD = candidateDocsRDD.mapToPair(doc -> {
            double score = 0.0;
            for (String term : queryTerms) {
                if (tfidfMatrix.containsKey(term)) {
                    score += tfidfMatrix.get(term).get(doc);
                }
            }
            return new Tuple2<>(doc, score);
        });
        
        // Sort by score and collect
        List<Tuple2<String, Double>> rankedDocs = scoresRDD
                .sortBy(tuple -> tuple._2(), false, 1)
                .collect();
        
        System.out.println("Matching Documents (ranked by similarity):");
        for (int i = 0; i < rankedDocs.size(); i++) {
            Tuple2<String, Double> docScore = rankedDocs.get(i);
            System.out.printf("%d. Doc%s (Score: %.4f)\n", i + 1, docScore._1(), docScore._2());
        }
    }
    
    // Helper methods (reused from Main class)
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
}