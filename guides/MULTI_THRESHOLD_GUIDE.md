# Multi-Threshold Deduplication Guide

## What is Multi-Threshold Mode?

Instead of running the deduplication multiple times with different thresholds, you can now create multiple outputs in **ONE RUN** by reusing the expensive embedding computation.

## How It Works

### Traditional Approach (Slow):
```bash
# Run 1: Threshold 0.85
python semantic_dedup.py data.json -o output1.json -t 0.85
# Time: ~90 minutes (60 min embeddings + 20 min dedup + 10 min save)

# Run 2: Threshold 0.80
python semantic_dedup.py data.json -o output2.json -t 0.80
# Time: ~90 minutes (60 min embeddings + 20 min dedup + 10 min save)

# Run 3: Threshold 0.90
python semantic_dedup.py data.json -o output3.json -t 0.90
# Time: ~90 minutes (60 min embeddings + 20 min dedup + 10 min save)

# TOTAL: 270 minutes (4.5 hours!)
```

### **New Multi-Threshold Approach (Fast):**
```bash
# One run with 3 thresholds
python semantic_dedup.py data.json -t 0.85,0.80,0.90

# Breakdown:
# - Load model: ~30 sec
# - Load data: ~5 sec
# - Create embeddings: ~60 min (DONE ONCE!)
# - Build FAISS index: ~1 min (DONE ONCE!)
# - Find duplicates (0.85): ~20 min
# - Save output 1: ~15 sec
# - Find duplicates (0.80): ~20 min (reuses embeddings!)
# - Save output 2: ~15 sec
# - Find duplicates (0.90): ~20 min (reuses embeddings!)
# - Save output 3: ~15 sec

# TOTAL: ~122 minutes (2 hours!)
# SAVES: 148 minutes (2.5 hours!)
```

## Usage Examples

### Basic Multi-Threshold
```bash
python semantic_dedup.py data.json -t 0.85,0.80,0.90
```

**Creates:**
- `data_dedup_t0.85.json` - Conservative (keeps more records)
- `data_dedup_t0.8.json` - Moderate
- `data_dedup_t0.9.json` - Aggressive (removes more duplicates)

### With Custom Output Path
```bash
python semantic_dedup.py data.json -o clean.json -t 0.85,0.80,0.90
```

**Creates:**
- `clean_t0.85.json`
- `clean_t0.8.json`
- `clean_t0.9.json`

### With Different Format
```bash
python semantic_dedup.py data.csv -o output.csv -t 0.85,0.80,0.75 -f csv
```

**Creates:**
- `output_t0.85.csv`
- `output_t0.8.csv`
- `output_t0.75.csv`

### Many Thresholds (Find the Perfect One)
```bash
python semantic_dedup.py data.json -t 0.95,0.90,0.85,0.80,0.75,0.70
```

**Creates 6 outputs** with different levels of deduplication!

## Understanding the Output

### Example Run:
```bash
python semantic_dedup.py converted_data.json -t 0.90,0.85,0.80
```

**Console Output:**
```
============================================================
MULTI-THRESHOLD MODE: Running with 3 thresholds
Thresholds: 0.9, 0.85, 0.8
============================================================

Loading embedding model: intfloat/multilingual-e5-base...
Model loaded successfully! Embedding dimension: 768
Loading data from converted_data.json...
Loaded 43894 records

============================================================
PHASE 1/3: Creating embeddings (only done once!)
============================================================
Creating embeddings...
Batches: 100%|████████████████████| 1372/1372 [1:02:34<00:00, 2.74s/it]

============================================================
PHASE 2/3: Building FAISS index (only done once!)
============================================================
Building FAISS index...

============================================================
PHASE 3/3: Finding duplicates for 3 thresholds
============================================================

────────────────────────────────────────────────────────────
Processing threshold 1/3: 0.9
────────────────────────────────────────────────────────────
Finding duplicates...
Processing records: 100%|████████████| 43894/43894 [18:23<00:00, 39.77it/s]

Found 8234 duplicates across 4521 groups

Results for threshold 0.9:
  Original records: 43894
  Deduplicated records: 35660
  Removed: 8234 (18.8%)
  Saved to: converted_data_dedup_t0.9.json

────────────────────────────────────────────────────────────
Processing threshold 2/3: 0.85
────────────────────────────────────────────────────────────
Finding duplicates...
Processing records: 100%|████████████| 43894/43894 [19:12<00:00, 38.09it/s]

Found 12456 duplicates across 6234 groups

Results for threshold 0.85:
  Original records: 43894
  Deduplicated records: 31438
  Removed: 12456 (28.4%)
  Saved to: converted_data_dedup_t0.85.json

────────────────────────────────────────────────────────────
Processing threshold 3/3: 0.8
────────────────────────────────────────────────────────────
Finding duplicates...
Processing records: 100%|████████████| 43894/43894 [20:01<00:00, 36.54it/s]

Found 15892 duplicates across 7823 groups

Results for threshold 0.8:
  Original records: 43894
  Deduplicated records: 28002
  Removed: 15892 (36.2%)
  Saved to: converted_data_dedup_t0.8.json

============================================================
ALL DONE! Processed 3 thresholds successfully
============================================================
```

## Choosing the Right Thresholds

### Understanding Similarity Thresholds:

| Threshold | Meaning | Use Case | Typical Removal % |
|-----------|---------|----------|-------------------|
| **0.95** | Very strict | Only remove near-identical copies | 5-15% |
| **0.90** | Strict | Remove obvious duplicates | 15-25% |
| **0.85** | Balanced | Good balance (default) | 25-35% |
| **0.80** | Moderate | More aggressive deduplication | 35-45% |
| **0.75** | Aggressive | Remove similar content | 45-60% |
| **0.70** | Very aggressive | May remove legitimate variations | 60-75% |

### Recommended Combinations:

**Conservative approach (keep variations):**
```bash
-t 0.90,0.85
```

**Balanced approach (most common):**
```bash
-t 0.90,0.85,0.80
```

**Exploratory approach (find the sweet spot):**
```bash
-t 0.95,0.90,0.85,0.80,0.75
```

**Aggressive cleaning:**
```bash
-t 0.85,0.80,0.75,0.70
```

## Performance Analysis

### Time Breakdown (43,894 records):

| Phase | Single Run | 3 Separate Runs | Multi-Threshold (3) | Savings |
|-------|-----------|-----------------|---------------------|---------|
| Load Model | 30 sec | 90 sec | 30 sec | 60 sec |
| Load Data | 5 sec | 15 sec | 5 sec | 10 sec |
| **Create Embeddings** | **60 min** | **180 min** | **60 min** | **120 min** |
| Build FAISS Index | 1 min | 3 min | 1 min | 2 min |
| Find Duplicates | 20 min | 60 min | 60 min | 0 min |
| Save Results | 15 sec | 45 sec | 45 sec | 0 sec |
| **TOTAL** | **~82 min** | **~244 min** | **~122 min** | **~122 min** |

**Time savings increase with more thresholds:**
- 2 thresholds: Save ~50% time
- 3 thresholds: Save ~50% time
- 5 thresholds: Save ~60% time
- 10 thresholds: Save ~70% time

## Use Cases

### 1. Finding the Optimal Threshold
You're not sure what threshold works best for your data:
```bash
python semantic_dedup.py data.json -t 0.95,0.90,0.85,0.80,0.75
```

Then manually inspect the outputs to see which gives the best balance.

### 2. Creating Multiple Datasets for Different Purposes
```bash
# Conservative for training data (keep variations)
# Aggressive for evaluation data (remove all duplicates)
python semantic_dedup.py data.json -t 0.90,0.75
```

### 3. A/B Testing
Create different versions to test which works better:
```bash
python semantic_dedup.py dataset.csv -t 0.85,0.80 -f csv
```

### 4. Quality vs Quantity Trade-off
```bash
# Create 3 versions: high quality (fewer items), balanced, high quantity
python semantic_dedup.py data.jsonl -t 0.90,0.85,0.80
```

## Tips and Tricks

### 1. Start Wide, Then Narrow
```bash
# First run: Test a wide range
python semantic_dedup.py data.json -t 0.95,0.90,0.85,0.80,0.75

# After inspection, narrow down:
python semantic_dedup.py data.json -t 0.87,0.85,0.83
```

### 2. Combine with Different Models
```bash
# Compare model performance at different thresholds
python semantic_dedup.py data.json -t 0.90,0.85,0.80 -m intfloat/multilingual-e5-base
python semantic_dedup.py data.json -t 0.90,0.85,0.80 -m intfloat/multilingual-e5-large
```

### 3. Incremental Thresholds for Fine-Tuning
```bash
# Very fine-grained comparison
python semantic_dedup.py data.json -t 0.88,0.86,0.84,0.82
```

## Limitations

1. **Memory Usage:** All deduplicated dataframes are held in memory during processing. For very large datasets (>1M records) with many thresholds, this might use significant RAM.

2. **Disk Space:** Creating multiple outputs requires disk space. 10 thresholds = 10 output files.

3. **Duplicate Detection Time:** While embeddings are reused, the duplicate detection still runs for each threshold. This scales linearly with the number of thresholds.

## Single Threshold Mode (Original Behavior)

The original single-threshold mode still works exactly as before:

```bash
# Single threshold - original behavior
python semantic_dedup.py data.json -o output.json -t 0.85
```

This creates just one output file: `output.json`

## Technical Details

### What Gets Reused?
✅ **Model loading** - Done once
✅ **Data loading** - Done once
✅ **Embedding creation** - Done once (the expensive part!)
✅ **FAISS index building** - Done once

### What Gets Repeated?
🔄 **Duplicate detection** - Once per threshold (but fast with FAISS)
🔄 **File saving** - Once per threshold (I/O operation)

### Code Workflow:
```
1. Load model (once)
2. Load data (once)
3. Create embeddings (once) ← THE EXPENSIVE PART
4. Build FAISS index (once)
5. For each threshold:
   a. Set threshold
   b. Find duplicates (reuses index)
   c. Create deduplicated dataframe
   d. Save to file with threshold in filename
```

## Conclusion

Multi-threshold mode is a huge time-saver when you need to:
- Experiment with different threshold values
- Create multiple datasets for different purposes
- Find the optimal deduplication level
- Compare results across different settings

**The key insight:** Embedding creation is ~70% of the total time. By doing it once and reusing the results, you save massive amounts of time when testing multiple thresholds!
