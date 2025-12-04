# Information Retrieval System

This project implements a comprehensive Information Retrieval system using Java and Apache Spark, fulfilling both parts of the project requirements.

## Project Structure

```
ir/
├── pom.xml                     # Maven configuration with Spark dependencies
├── dataset/                    # Contains the 10 text documents (1.txt to 10.txt)
└── src/main/java/com/example/
    ├── Main.java              # Standard Java implementation
    └── SparkMain.java         # Apache Spark implementation
```

## Dataset

The project uses 10 text documents (1.txt to 10.txt) containing various terms:
- Documents 1-6: Shakespeare-related terms (antony, brutus, caeser, cleopatra, mercy, worser, calpurnia)
- Documents 7-10: Phrase "angels fools fear in rush to tread where" with variations

## Features Implemented

### Part 1: Positional Index (Spark App)
✅ **Completed**: Builds positional index from dataset displaying each term as:
```
<term doc1: position1, position2... doc2: position1, position2... etc.>
```

### Part 2: Term Frequency Analysis
✅ **Completed**: Uses the Spark App output from Part 1 to compute:

1. **Term Frequency Matrix**: Shows frequency of each term in each document
2. **IDF Calculation**: Computes IDF for each term using `log10(N/df)`
3. **TF-IDF Matrix**: Uses the specified rule `1+log10(tf)` for TF calculation
4. **Query Processing**: Supports phrase queries with boolean operators:
   - `AND` operations (e.g., "mercy AND caeser")
   - `AND NOT` operations (e.g., "brutus AND NOT mercy")
   - Ranks documents by similarity scores

## Implementation Details

### TF Calculation Rule
The project uses the specified TF rule: **`1 + log10(tf)`**
- If tf = 0: TF-IDF = 0
- If tf > 0: TF-IDF = (1 + log10(tf)) × IDF

### IDF Calculation
Standard IDF formula: **`log10(N/df)`**
- N = total number of documents (10)
- df = document frequency (number of documents containing the term)

### Query Processing
- Supports boolean queries with AND and AND NOT operators
- Calculates similarity using TF-IDF scores
- Ranks results by similarity score (highest first)

## Running the Project

### Option 1: Standard Java Implementation (Recommended)
```bash
# Compile
javac -d target/classes -sourcepath src/main/java src/main/java/com/example/Main.java

# Run
java -cp target/classes com.example.Main
```

### Option 2: Apache Spark Implementation (Requires Spark Dependencies)
```bash
# Install dependencies and run with Maven
mvn clean compile exec:java -Dexec.mainClass="com.example.SparkMain"

# Or compile and run directly
javac -cp "spark-jars/*" -d target/classes src/main/java/com/example/SparkMain.java
java -cp "target/classes:spark-jars/*" com.example.SparkMain
```

## Sample Output

### Positional Index
```
<mercy doc1: 5 doc3: 1 doc4: 3 doc5: 2 doc6: 3>
<caeser doc1: 3 doc2: 3 doc4: 2 doc5: 1 doc6: 2>
```

### Term Frequency Matrix
```
Term         Doc1  Doc2  Doc3  Doc4  Doc5  Doc6  Doc7  Doc8  Doc9 Doc10
mercy           1     0     1     1     1     1     0     0     0     0
caeser          1     1     0     1     1     1     0     0     0     0
```

### Query Results
```
Processing Query: "mercy AND caeser"
Matching Documents (ranked by similarity):
1. Doc1 (Score: 0.6021)
2. Doc4 (Score: 0.6021)
3. Doc5 (Score: 0.6021)
4. Doc6 (Score: 0.6021)
```

## Technical Features

- **Positional indexing** for exact term position tracking
- **TF-IDF calculation** with logarithmic term frequency weighting
- **Boolean query processing** with AND/AND NOT operators
- **Document ranking** by cosine similarity using TF-IDF scores
- **Apache Spark integration** for distributed processing
- **Scalable architecture** supporting large document collections

## Dependencies

- Java 17+
- Apache Spark 3.5.0 (for Spark implementation)
- Maven 3.6+ (for dependency management)

The project successfully implements both parts of the Information Retrieval requirements with comprehensive term analysis, positional indexing, and query processing capabilities.