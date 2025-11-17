# Semantic Deduplication Tool

Fast and efficient semantic deduplication using sentence embeddings and FAISS indexing.

## Features

- **Fast FAISS indexing** - Significantly faster than naive O(n²) similarity search
- **Local embedding model** - Uses sentence-transformers (runs on your 6GB GPU)
- **Multiple formats** - Supports JSON, JSONL, CSV, TSV, and Parquet
- **Configurable threshold** - Adjustable similarity threshold
- **Smart duplicate handling** - Keeps the longest record when duplicates are found
- **Whole-record comparison** - Compares entire records for semantic similarity

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

Basic usage:
```bash
python semantic_dedup.py input.json -o output.json
```

The first run will download the embedding model (~1GB). Subsequent runs will be faster.

## Usage Examples

### Deduplicate a JSON file
```bash
python semantic_dedup.py data.json -o data_clean.json
```

### Deduplicate a CSV file with custom threshold
```bash
python semantic_dedup.py data.csv -o data_clean.csv -t 0.9
```

### Convert format while deduplicating
```bash
python semantic_dedup.py input.json -o output.csv -f csv
```

### Use a different embedding model
```bash
python semantic_dedup.py data.jsonl -o clean.jsonl -m all-mpnet-base-v2
```

### Auto-generate output filename
```bash
python semantic_dedup.py data.json
# Creates: data_dedup.json
```

### **NEW: Multiple thresholds in ONE run!**
```bash
# Create 3 outputs with different thresholds (0.85, 0.80, 0.90)
# Only creates embeddings ONCE - saves tons of time!
python semantic_dedup.py data.json -t 0.85,0.80,0.90

# Creates:
#   data_dedup_t0.85.json  (conservative - keeps more)
#   data_dedup_t0.8.json   (moderate)
#   data_dedup_t0.9.json   (aggressive - removes more)
```

**Time savings example (43k records):**
- Running 3 times separately: ~90 min × 3 = **270 minutes**
- Running once with 3 thresholds: 60 min (embeddings) + 20 min × 3 (dedup) = **120 minutes**
- **Saves: 150 minutes!** (2.5 hours)

## Command Line Arguments

```
positional arguments:
  input                 Input file path

optional arguments:
  -h, --help            Show help message
  -o, --output          Output file path (default: input_dedup.ext)
  -t, --threshold       Similarity threshold(s) for duplicates (0-1, default: 0.85)
                        Can be single value (0.85) or multiple comma-separated (0.85,0.80,0.90)
  -m, --model          Sentence-transformer model name (default: intfloat/multilingual-e5-base)
  -f, --format         Output format: json, jsonl, csv, tsv, parquet
```

## Supported File Formats

- **JSON** - `.json` files
- **JSON Lines** - `.jsonl` files
- **CSV** - `.csv` files
- **TSV** - `.tsv` or `.txt` files
- **Parquet** - `.parquet` files

## How It Works

1. **Load Data** - Reads your dataset from the input file
2. **Create Embeddings** - Converts each record to a semantic vector using sentence-transformers
3. **Build FAISS Index** - Creates an efficient similarity search index
4. **Find Duplicates** - Uses cosine similarity to identify duplicate records
5. **Keep Longest** - When duplicates are found, keeps the record with the most content
6. **Save Results** - Writes deduplicated dataset to output file

## Embedding Models

### Recommended for Turkish (Best to Good):

1. **`intfloat/multilingual-e5-base`** (Default)
   - Size: ~1GB, 768-dimensional embeddings
   - Excellent quality and multilingual support
   - Supports 100 languages including Turkish
   - Fits easily in 6GB VRAM
   - **Recommended for most use cases**

2. **`paraphrase-multilingual-mpnet-base-v2`**
   - Size: ~420MB, 768-dimensional embeddings
   - Good balance of quality and speed for Turkish
   - Supports 50+ languages including Turkish
   - Lighter alternative to e5-base

3. **`intfloat/multilingual-e5-large`**
   - Size: ~2.5GB, 1024-dimensional embeddings
   - Higher quality multilingual embeddings
   - Excellent Turkish support
   - Still fits comfortably in 6GB VRAM

4. **`emrecan/bert-base-turkish-cased-mean-nli-stsb-tr`**
   - Size: ~500MB, 768-dimensional
   - Turkish-specific model
   - Trained on Turkish NLI and STS datasets
   - Good for Turkish-only datasets

5. **`Trendyol/TY-ecomm-embed-multilingual-base-v1.2.0`**
   - Turkish e-commerce company's model
   - Optimized for Turkish + multilingual text
   - Good for product descriptions, reviews

### For comparison (smaller, lower quality):
- `all-MiniLM-L6-v2` - Fast, lightweight (~80MB) but English-focused
- `paraphrase-multilingual-MiniLM-L12-v2` - Smaller multilingual option (~420MB, 384-dim)

## Similarity Threshold

The threshold determines how similar records must be to be considered duplicates:

- **0.95+** - Very strict, only near-identical items
- **0.85-0.90** - Balanced (recommended)
- **0.75-0.80** - Aggressive deduplication
- **<0.75** - Very aggressive, may remove distinct items

## Performance

FAISS indexing makes this tool scale efficiently:
- **1K records** - Few seconds
- **10K records** - ~1 minute
- **100K records** - ~10-15 minutes
- **1M+ records** - Scales well, minutes to hours depending on data

## Example Output

```
Loading embedding model: intfloat/multilingual-e5-base...
First run will download the model (~1GB for e5-base, ~420MB for mpnet)
Model will be cached for future use...
Model loaded successfully! Embedding dimension: 768
Loading data from data.json...
Loaded 10000 records
Creating embeddings...
100%|████████████████████| 10000/10000 [00:25<00:00, 392.16it/s]
Building FAISS index...
Finding duplicates...
Processing records: 100%|████████| 10000/10000 [00:42<00:00, 236.51it/s]

Found 2341 duplicates across 1876 groups

Deduplication complete:
  Original records: 10000
  Deduplicated records: 7659
  Removed: 2341 (23.4%)
Saving deduplicated data to data_clean.json...
Saved to data_clean.json
```

## GPU Usage

The tool automatically uses your GPU if available. The embedding model will run on your 6GB NVIDIA card, providing faster processing.

## Troubleshooting

**Out of memory?**
- Reduce batch size in the code (line with `batch_size=32`)
- Use a smaller model like `all-MiniLM-L6-v2`

**Too slow?**
- Use a smaller/faster model
- Reduce the number of similar items searched (line with `k = min(100, n_records)`)

**Not finding duplicates?**
- Lower the similarity threshold (e.g., `-t 0.8`)

**Finding too many duplicates?**
- Raise the similarity threshold (e.g., `-t 0.9`)
