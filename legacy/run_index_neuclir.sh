#!/bin/sh
#SBATCH --job-name=bm25
#SBATCH --cpus-per-task=32
#SBATCH --partition cpu
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=%x.out

module load anaconda3/2024.2
conda activate basic

mkdir -p temp
# python3 convert_to_beir_format.py

python -m pyserini.index.lucene \
    --collection BeirFlatCollection \
    --input temp/ \
    --index /exp/jhueiju/neuclir/index/title+text.mlir.mt.lucene \
    --generator DefaultLuceneDocumentGenerator \
    --threads 128
