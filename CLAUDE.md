# Changelog

All notable changes to the Semantic Deduplication Tool.

## [1.1.0] - 2025-01-10

### Added
- **Multi-Threshold Mode**: Process multiple similarity thresholds in a single run
  - Reuses embeddings and FAISS index across thresholds
  - Saves 50-70% processing time when testing multiple thresholds
  - Example: `python semantic_dedup.py data.json -t 0.85,0.80,0.90`
  - Automatically generates separate output files with threshold in filename

- **Enhanced Terminal Output**:
  - Displays current embedding model name during processing
  - Shows device being used (CPU or CUDA)
  - Shows batch size and record count information
  - Progress indicators for multi-threshold processing

- **Documentation**:
  - Added `MULTI_THRESHOLD_GUIDE.md` with comprehensive usage examples
  - Added `TECHNICAL_DOCUMENTATION.md` explaining workflow and FAISS internals
  - Updated `README.md` with multi-threshold examples and time savings

### Changed
- Default embedding model changed from `all-MiniLM-L6-v2` to `paraphrase-multilingual-mpnet-base-v2`
  - Better multilingual support (50+ languages)
  - Improved Turkish language performance
  - Higher quality embeddings (768-dim vs 384-dim)

- Threshold parameter now accepts comma-separated values for multi-threshold mode
- Output filenames now include threshold value when using multi-threshold mode

### Performance
- Multi-threshold mode performance (43,894 records example):
  - 3 separate runs: ~270 minutes
  - Multi-threshold (3): ~122 minutes
  - Time savings: ~148 minutes (55% faster)

## [1.0.0] - 2025-01-10

### Initial Release
- Semantic deduplication using sentence embeddings
- FAISS indexing for efficient similarity search
- Support for multiple file formats (JSON, JSONL, CSV, TSV, Parquet)
- Configurable similarity threshold
- Keeps longest record when duplicates found
- Turkish language support with multilingual models
