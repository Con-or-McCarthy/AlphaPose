#!/usr/bin/env bash
set -euo pipefail

# Run AlphaPose on image directories in chunks to avoid long-run memory spikes.
#
# Usage:
#   bash scripts/inference_chunked.sh \
#     <cfg> <checkpoint> <input_dir> <output_dir> [chunk_size]
#
# Example:
#   bash scripts/inference_chunked.sh \
#     configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml \
#     pretrained_models/halpe26_fast_res50_256x192.pth \
#     /path/to/frames_1fps \
#     /path/to/alphapose_1fps \
#     400

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <cfg> <checkpoint> <input_dir> <output_dir> [chunk_size]"
  exit 1
fi

CONFIG="$1"
CKPT="$2"
INPUT_DIR="$3"
OUTDIR="$4"
CHUNK_SIZE="${5:-400}"

if [[ ! -d "$INPUT_DIR" ]]; then
  echo "Input directory does not exist: $INPUT_DIR"
  exit 1
fi

mkdir -p "$OUTDIR"
mkdir -p "$OUTDIR/lists"
mkdir -p "$OUTDIR/json_chunks"

FILELIST="$OUTDIR/lists/all_images.txt"

python - "$INPUT_DIR" "$FILELIST" <<'PY'
import os
import sys
from pathlib import Path

indir = Path(sys.argv[1])
outfile = Path(sys.argv[2])

exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
files = [p for p in indir.iterdir() if p.is_file() and p.suffix.lower() in exts]
files.sort(key=lambda p: p.name)

if not files:
    print(f"No images found in {indir}")
    sys.exit(2)

outfile.write_text("".join(f"{p.name}\n" for p in files))
print(f"Wrote {len(files)} images to {outfile}")
PY

TOTAL=$(python - "$FILELIST" <<'PY'
import sys
from pathlib import Path
txt = Path(sys.argv[1]).read_text().splitlines()
print(len(txt))
PY
)

echo "Total frames: $TOTAL"
echo "Chunk size: $CHUNK_SIZE"

START=0
while [[ "$START" -lt "$TOTAL" ]]; do
  END=$((START + CHUNK_SIZE))
  if [[ "$END" -gt "$TOTAL" ]]; then
    END="$TOTAL"
  fi

  CHUNK_LIST="$OUTDIR/lists/chunk_${START}_${END}.txt"
  DONE_FLAG="$OUTDIR/lists/chunk_${START}_${END}.done"
  CHUNK_JSON_REL="json_chunks/alphapose-results_${START}_${END}.json"
  CHUNK_JSON_ABS="$OUTDIR/$CHUNK_JSON_REL"

  if [[ -f "$DONE_FLAG" && -f "$CHUNK_JSON_ABS" ]]; then
    echo "Skipping chunk ${START}:${END} (already done)"
    START="$END"
    continue
  fi

  python - "$FILELIST" "$CHUNK_LIST" "$START" "$END" <<'PY'
import sys
from pathlib import Path

all_list = Path(sys.argv[1]).read_text().splitlines()
chunk_list = Path(sys.argv[2])
start = int(sys.argv[3])
end = int(sys.argv[4])

chunk = all_list[start:end]
chunk_list.write_text("".join(f"{x}\n" for x in chunk))
print(f"Prepared {len(chunk)} files in {chunk_list}")
PY

  echo "Running chunk ${START}:${END} ..."
  python scripts/demo_inference.py \
    --cfg "$CONFIG" \
    --checkpoint "$CKPT" \
    --indir "$INPUT_DIR" \
    --list "$CHUNK_LIST" \
    --save_img \
    --save_mask \
    --save_mask_vis \
    --vis_fast \
    --sp \
    --detbatch 1 \
    --posebatch 16 \
    --qsize 16 \
    --outdir "$OUTDIR" \
    --outputfile "$CHUNK_JSON_REL"

  touch "$DONE_FLAG"
  echo "Finished chunk ${START}:${END}"
  START="$END"
done

FINAL_JSON="$OUTDIR/alphapose-results.json"
python - "$OUTDIR/json_chunks" "$FINAL_JSON" <<'PY'
import json
import sys
from pathlib import Path

chunk_dir = Path(sys.argv[1])
final_json = Path(sys.argv[2])

chunk_files = sorted(chunk_dir.glob("alphapose-results_*.json"))
if not chunk_files:
    print(f"No chunk json files found in {chunk_dir}")
    sys.exit(3)

merged = []
for p in chunk_files:
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        print(f"Skipping non-list json: {p}")
        continue
    merged.extend(data)

with final_json.open("w", encoding="utf-8") as f:
    json.dump(merged, f)

print(f"Merged {len(chunk_files)} chunks with {len(merged)} detections -> {final_json}")
PY

echo "All chunks complete. Outputs are in: $OUTDIR"
echo "Merged keypoints JSON: $FINAL_JSON"
