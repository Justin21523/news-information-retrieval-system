#!/bin/bash
# CNIRS Complete Indexing Pipeline
#
# This script completes the indexing pipeline after preprocessing:
# 1. Build search indexes from preprocessed data
# 2. Generate BERT embeddings
# 3. Update app configuration
#
# Usage:
#   bash scripts/complete_indexing.sh
#
# Prerequisites:
#   - Preprocessing completed (data/preprocessed/merged_14days_preprocessed.jsonl exists)
#   - Python environment: conda activate ai_env

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

PYTHON="/home/justin/miniconda3/envs/ai_env/bin/python"
INPUT_FILE="data/preprocessed/merged_14days_preprocessed.jsonl"
OUTPUT_DIR="data/indexes_10k"

echo "============================================================"
echo "CNIRS Complete Indexing Pipeline"
echo "============================================================"
echo "Project root: $PROJECT_ROOT"
echo "Input: $INPUT_FILE"
echo "Output: $OUTPUT_DIR"
echo ""

# Check input file
if [ ! -f "$INPUT_FILE" ]; then
    echo "ERROR: Input file not found: $INPUT_FILE"
    echo "Please run preprocessing first:"
    echo "  python scripts/preprocess_batch.py"
    exit 1
fi

# Get record count
RECORD_COUNT=$(wc -l < "$INPUT_FILE")
echo "Records found: $RECORD_COUNT"
echo ""

# Step 1: Build search indexes
echo "============================================================"
echo "Step 1: Building Search Indexes"
echo "============================================================"
$PYTHON scripts/build_indexes_from_preprocessed.py \
    --input "$INPUT_FILE" \
    --output "$OUTPUT_DIR"

# Verify indexes created
echo ""
echo "Verifying indexes..."
ls -la "$OUTPUT_DIR/"

# Step 2: Build BERT embeddings
echo ""
echo "============================================================"
echo "Step 2: Building BERT Embeddings"
echo "============================================================"
$PYTHON scripts/build_bert_embeddings.py \
    --input "$INPUT_FILE" \
    --output "$OUTPUT_DIR/bert_embeddings.npy" \
    --model paraphrase-multilingual-MiniLM-L12-v2 \
    --batch-size 64

# Verify BERT embeddings
echo ""
echo "Verifying BERT embeddings..."
ls -la "$OUTPUT_DIR/bert_embeddings.*"

# Step 3: Update app configuration (backup first)
echo ""
echo "============================================================"
echo "Step 3: Updating App Configuration"
echo "============================================================"

# Backup original
if [ -f "app_simple.py" ]; then
    cp app_simple.py app_simple.py.backup
    echo "Created backup: app_simple.py.backup"
fi

# Update INDEX_DIR
sed -i "s|app.config\['INDEX_DIR'\] = project_root / 'data' / 'indexes'|app.config['INDEX_DIR'] = project_root / 'data' / 'indexes_10k'|g" app_simple.py

# Update preprocessed file path
sed -i "s|preprocessed_file = project_root / 'data' / 'preprocessed' / 'cna_mvp_preprocessed.jsonl'|preprocessed_file = project_root / 'data' / 'preprocessed' / 'merged_14days_preprocessed.jsonl'|g" app_simple.py

echo "App configuration updated."

# Final summary
echo ""
echo "============================================================"
echo "INDEXING COMPLETE!"
echo "============================================================"
echo "Index directory: $OUTPUT_DIR"
echo "Documents indexed: $RECORD_COUNT"
echo ""
echo "Index files:"
ls -la "$OUTPUT_DIR/"
echo ""
echo "To start the application:"
echo "  python app_simple.py"
echo ""
echo "To verify:"
echo "  curl http://localhost:5001/api/stats"
echo "============================================================"
