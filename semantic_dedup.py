#!/usr/bin/env python3
"""
Semantic Deduplication Tool
Uses sentence embeddings and FAISS indexing for efficient similarity search
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm


class SemanticDeduplicator:
    """Handles semantic deduplication using embeddings and FAISS"""

    def __init__(self, model_name: str = 'paraphrase-multilingual-mpnet-base-v2', similarity_threshold: float = 0.85):
        """
        Initialize the deduplicator

        Args:
            model_name: Name of the sentence-transformer model to use
            similarity_threshold: Threshold for considering items as duplicates (0-1)
        """
        print(f"Loading embedding model: {model_name}...")
        print("First run will download the model (~420MB for mpnet, ~80MB for MiniLM)")
        print("Model will be cached for future use...")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.dimension = self.model.get_sentence_embedding_dimension()

        # Check device (CPU or GPU)
        device = self.model.device
        print(f"Model loaded successfully! Embedding dimension: {self.dimension}")
        print(f"Running on: {device}")

    def load_data(self, file_path: str) -> Tuple[pd.DataFrame, str]:
        """
        Load data from various file formats

        Args:
            file_path: Path to the input file

        Returns:
            Tuple of (DataFrame, file_format)
        """
        file_ext = Path(file_path).suffix.lower()

        print(f"Loading data from {file_path}...")

        if file_ext == '.json':
            df = pd.read_json(file_path)
            return df, 'json'
        elif file_ext == '.jsonl':
            df = pd.read_json(file_path, lines=True)
            return df, 'jsonl'
        elif file_ext == '.csv':
            df = pd.read_csv(file_path)
            return df, 'csv'
        elif file_ext in ['.tsv', '.txt']:
            df = pd.read_csv(file_path, sep='\t')
            return df, 'tsv'
        elif file_ext == '.parquet':
            df = pd.read_parquet(file_path)
            return df, 'parquet'
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")

    def record_to_text(self, record: Dict[str, Any]) -> str:
        """
        Convert a record (row) to text for embedding

        Args:
            record: Dictionary representing a single record

        Returns:
            Concatenated text representation
        """
        # Concatenate all fields with their keys for context
        parts = []
        for key, value in record.items():
            if pd.notna(value):  # Skip NaN values
                parts.append(f"{key}: {str(value)}")
        return " | ".join(parts)

    def create_embeddings(self, df: pd.DataFrame) -> np.ndarray:
        """
        Create embeddings for all records

        Args:
            df: DataFrame containing the records

        Returns:
            Numpy array of embeddings
        """
        print(f"Creating embeddings using model: {self.model_name}")
        print(f"Processing {len(df)} records with batch_size=32")
        texts = []
        for _, row in df.iterrows():
            text = self.record_to_text(row.to_dict())
            texts.append(text)

        # Encode in batches for efficiency
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32,
            convert_to_numpy=True
        )

        return embeddings

    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.IndexFlatIP:
        """
        Build FAISS index for efficient similarity search
        Uses Inner Product (IP) index with normalized vectors for cosine similarity

        Args:
            embeddings: Numpy array of embeddings

        Returns:
            FAISS index
        """
        print("Building FAISS index...")

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)

        # Create Inner Product index (equivalent to cosine similarity with normalized vectors)
        index = faiss.IndexFlatIP(self.dimension)
        index.add(embeddings)

        return index

    def find_duplicates(self, df: pd.DataFrame, embeddings: np.ndarray,
                       index: faiss.IndexFlatIP) -> List[int]:
        """
        Find duplicate records using FAISS similarity search

        Args:
            df: DataFrame containing the records
            embeddings: Numpy array of embeddings
            index: FAISS index

        Returns:
            List of indices to keep
        """
        print("Finding duplicates...")

        n_records = len(df)
        keep_indices = set(range(n_records))
        duplicate_groups = []

        # For each record, find similar ones
        for i in tqdm(range(n_records), desc="Processing records"):
            if i not in keep_indices:
                continue

            # Search for similar items (k+1 because the item itself will be included)
            # We search for more items to find all potential duplicates
            k = min(100, n_records)  # Limit search to avoid performance issues
            distances, indices = index.search(embeddings[i:i+1], k)

            # Find duplicates above threshold
            duplicates = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx != i and dist >= self.similarity_threshold and idx in keep_indices:
                    duplicates.append(idx)

            # If duplicates found, keep the longest one
            if duplicates:
                all_items = [i] + duplicates

                # Calculate text lengths for all items
                lengths = []
                for idx in all_items:
                    text = self.record_to_text(df.iloc[idx].to_dict())
                    lengths.append(len(text))

                # Keep the item with the longest text
                keep_idx = all_items[np.argmax(lengths)]

                # Remove all others from keep_indices
                for idx in all_items:
                    if idx != keep_idx and idx in keep_indices:
                        keep_indices.remove(idx)

                if len(duplicates) > 0:
                    duplicate_groups.append({
                        'kept': keep_idx,
                        'removed': [idx for idx in duplicates if idx != keep_idx],
                        'similarity': float(distances[0][1]) if len(duplicates) > 0 else 1.0
                    })

        print(f"\nFound {n_records - len(keep_indices)} duplicates across {len(duplicate_groups)} groups")

        return sorted(list(keep_indices))

    def save_data(self, df: pd.DataFrame, output_path: str, file_format: str):
        """
        Save deduplicated data to file

        Args:
            df: DataFrame to save
            output_path: Path to save the file
            file_format: Format to save in
        """
        print(f"Saving deduplicated data to {output_path}...")

        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

        if file_format == 'json':
            df.to_json(output_path, orient='records', indent=2, force_ascii=False)
        elif file_format == 'jsonl':
            df.to_json(output_path, orient='records', lines=True, force_ascii=False)
        elif file_format == 'csv':
            df.to_csv(output_path, index=False)
        elif file_format == 'tsv':
            df.to_csv(output_path, sep='\t', index=False)
        elif file_format == 'parquet':
            df.to_parquet(output_path, index=False)
        else:
            raise ValueError(f"Unsupported output format: {file_format}")

    def deduplicate(self, input_path: str, output_path: str = None,
                   output_format: str = None) -> pd.DataFrame:
        """
        Main deduplication pipeline

        Args:
            input_path: Path to input file
            output_path: Path to output file (optional)
            output_format: Format for output file (optional, defaults to input format)

        Returns:
            Deduplicated DataFrame
        """
        # Load data
        df, input_format = self.load_data(input_path)
        original_count = len(df)
        print(f"Loaded {original_count} records")

        if original_count == 0:
            print("No records to process")
            return df

        # Create embeddings
        embeddings = self.create_embeddings(df)

        # Build FAISS index
        index = self.build_faiss_index(embeddings)

        # Find duplicates
        keep_indices = self.find_duplicates(df, embeddings, index)

        # Create deduplicated dataframe
        df_dedup = df.iloc[keep_indices].reset_index(drop=True)
        final_count = len(df_dedup)

        print(f"\nDeduplication complete:")
        print(f"  Original records: {original_count}")
        print(f"  Deduplicated records: {final_count}")
        print(f"  Removed: {original_count - final_count} ({(original_count - final_count) / original_count * 100:.1f}%)")

        # Save if output path provided
        if output_path:
            format_to_use = output_format or input_format
            self.save_data(df_dedup, output_path, format_to_use)
            print(f"Saved to {output_path}")

        return df_dedup


def main():
    parser = argparse.ArgumentParser(
        description='Semantic deduplication using embeddings and FAISS',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single threshold (default behavior)
  python semantic_dedup.py input.json -o output.json
  python semantic_dedup.py input.csv -o output.csv -t 0.9

  # Multiple thresholds (creates multiple outputs in ONE run!)
  python semantic_dedup.py input.json -t 0.85,0.80,0.90
  python semantic_dedup.py input.csv -o output.csv -t 0.85,0.80,0.75

  # Different model
  python semantic_dedup.py input.jsonl -o output.jsonl -m intfloat/multilingual-e5-large

  # Convert format
  python semantic_dedup.py input.json -o output.csv -f csv

Recommended models for Turkish:
  paraphrase-multilingual-mpnet-base-v2 (default, best balance)
  intfloat/multilingual-e5-base (best balance, 100 languages)
  intfloat/multilingual-e5-large (higher quality, slower)
  emrecan/bert-base-turkish-cased-mean-nli-stsb-tr (Turkish-specific)
        """
    )

    parser.add_argument('input', help='Input file path')
    parser.add_argument('-o', '--output', help='Output file path (default: input_dedup.ext)')
    parser.add_argument('-t', '--threshold', type=str, default='0.85',
                       help='Similarity threshold(s) for duplicates. Single value (e.g., 0.85) or multiple comma-separated values (e.g., 0.85,0.80,0.90)')
    parser.add_argument('-m', '--model', default='paraphrase-multilingual-mpnet-base-v2',
                       help='Sentence-transformer model name (default: paraphrase-multilingual-mpnet-base-v2)')
    parser.add_argument('-f', '--format', choices=['json', 'jsonl', 'csv', 'tsv', 'parquet'],
                       help='Output format (default: same as input)')

    args = parser.parse_args()

    # Parse threshold(s)
    try:
        thresholds = [float(t.strip()) for t in args.threshold.split(',')]
    except ValueError:
        print("Error: Threshold must be a number or comma-separated numbers (e.g., 0.85 or 0.85,0.80)")
        sys.exit(1)

    # Validate thresholds
    for threshold in thresholds:
        if not 0 <= threshold <= 1:
            print(f"Error: Threshold {threshold} must be between 0 and 1")
            sys.exit(1)

    # Multiple thresholds mode
    if len(thresholds) > 1:
        print(f"\n{'='*60}")
        print(f"MULTI-THRESHOLD MODE: Running with {len(thresholds)} thresholds")
        print(f"Thresholds: {', '.join([str(t) for t in thresholds])}")
        print(f"{'='*60}\n")

        # Run deduplication with multiple thresholds (reusing embeddings)
        try:
            # Initialize deduplicator (will use first threshold, but we'll override later)
            deduplicator = SemanticDeduplicator(
                model_name=args.model,
                similarity_threshold=thresholds[0]
            )

            # Load data
            df, input_format = deduplicator.load_data(args.input)
            original_count = len(df)
            print(f"Loaded {original_count} records\n")

            if original_count == 0:
                print("No records to process")
                sys.exit(0)

            # Create embeddings ONCE (the expensive part!)
            print("=" * 60)
            print("PHASE 1/3: Creating embeddings (only done once!)")
            print(f"Model: {args.model}")
            print("=" * 60)
            embeddings = deduplicator.create_embeddings(df)

            # Build FAISS index ONCE
            print("\n" + "=" * 60)
            print("PHASE 2/3: Building FAISS index (only done once!)")
            print("=" * 60)
            index = deduplicator.build_faiss_index(embeddings)

            # Run deduplication for each threshold
            print("\n" + "=" * 60)
            print(f"PHASE 3/3: Finding duplicates for {len(thresholds)} thresholds")
            print("=" * 60 + "\n")

            for i, threshold in enumerate(thresholds, 1):
                print(f"\n{'─'*60}")
                print(f"Processing threshold {i}/{len(thresholds)}: {threshold}")
                print(f"{'─'*60}")

                # Update threshold
                deduplicator.similarity_threshold = threshold

                # Find duplicates with this threshold
                keep_indices = deduplicator.find_duplicates(df, embeddings, index)

                # Create deduplicated dataframe
                df_dedup = df.iloc[keep_indices].reset_index(drop=True)
                final_count = len(df_dedup)

                print(f"\nResults for threshold {threshold}:")
                print(f"  Original records: {original_count}")
                print(f"  Deduplicated records: {final_count}")
                print(f"  Removed: {original_count - final_count} ({(original_count - final_count) / original_count * 100:.1f}%)")

                # Generate output path for this threshold
                input_path = Path(args.input)
                if args.output:
                    # User provided output, add threshold to it
                    output_path_obj = Path(args.output)
                    output_path = str(output_path_obj.parent / f"{output_path_obj.stem}_t{threshold}{output_path_obj.suffix}")
                else:
                    # Auto-generate with threshold
                    output_path = str(input_path.parent / f"{input_path.stem}_dedup_t{threshold}{input_path.suffix}")

                # Save
                format_to_use = args.format or input_format
                deduplicator.save_data(df_dedup, output_path, format_to_use)
                print(f"  Saved to: {output_path}")

            print(f"\n{'='*60}")
            print(f"ALL DONE! Processed {len(thresholds)} thresholds successfully")
            print(f"{'='*60}\n")

        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

    else:
        # Single threshold mode (original behavior)
        threshold = thresholds[0]

        # Generate output path if not provided
        if not args.output:
            input_path = Path(args.input)
            args.output = str(input_path.parent / f"{input_path.stem}_dedup{input_path.suffix}")

        # Run deduplication
        try:
            deduplicator = SemanticDeduplicator(
                model_name=args.model,
                similarity_threshold=threshold
            )
            deduplicator.deduplicate(
                input_path=args.input,
                output_path=args.output,
                output_format=args.format
            )
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)


if __name__ == '__main__':
    main()
