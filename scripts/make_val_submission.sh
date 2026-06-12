#!/usr/bin/env bash
#
# Build one Codabench-shaped submission.zip per model variant
# (graph / no_graph / no_zones) for the VAL matches game_18, game_24,
# game_47.
#
# For each model it:
#   1. runs offline inference on every (match, half) -> predictions JSON
#      in the new {frame, team, jersey_number, action_class, score} shape
#   2. validates the per-match schema (--validate-only + --report)
#   3. packages predictions/ -> submission.zip (single predictions.json)
#
# Edit the CONFIG block below to point at your actual checkpoints /
# visual-feature cache, then run:  bash scripts/make_val_submission.sh
#
set -euo pipefail

# --------------------------------------------------------------- CONFIG
# Per-variant checkpoint paths. Change these to wherever your trained
# models live (the repo convention is checkpoints/<variant>/best.pt).
declare -A CHECKPOINTS=(
  [graph]="checkpoints/graph/best.pt"
  [no_graph]="checkpoints/no_graph/best.pt"
  [no_zones]="checkpoints/no_zones/best.pt"
)

CONFIG="config.toml"                              # [pcbas] output_dir lives here
VISUAL_CACHE="data/pcbas/visual_features"         # precomputed DINOv2 cache
VISUAL_BACKBONE="dinov2_vits14"
DEVICE="cuda:0"                                   # use "cpu" if no GPU

MATCHES=(game_18 game_24 game_47)
HALVES=(H1 H2)

# Inference hyper-parameters (mirror configs/train/graph/main.toml [validation]).
WINDOW_SIZE=128
STRIDE=96
DECODE_THRESHOLD=0.5
NMS_MODE="per_player_class"
NMS_RADIUS=12

OUT_ROOT="val_submission"                         # everything lands under here
# ------------------------------------------------------------ END CONFIG

for variant in "${!CHECKPOINTS[@]}"; do
  ckpt="${CHECKPOINTS[$variant]}"
  preds_dir="${OUT_ROOT}/${variant}/predictions"
  report="${OUT_ROOT}/${variant}/report.json"
  zip_out="${OUT_ROOT}/${variant}/submission.zip"

  echo "============================================================"
  echo "Model variant: ${variant}   checkpoint: ${ckpt}"
  echo "============================================================"

  if [[ ! -f "${ckpt}" ]]; then
    echo "!! checkpoint not found: ${ckpt} -- skipping ${variant}" >&2
    continue
  fi

  mkdir -p "${preds_dir}"

  # 1) Offline inference: one JSON per (match, half).
  for match in "${MATCHES[@]}"; do
    for half in "${HALVES[@]}"; do
      out="${preds_dir}/${match}__${half}.json"
      echo "--- infer ${variant} ${match} ${half} ---"
      # A match may not have a given half; infer.py exits non-zero in
      # that case, so don't let it abort the whole run.
      if ! python scripts/infer.py offline \
          --model-variant "${variant}" \
          --checkpoint "${ckpt}" \
          --config "${CONFIG}" \
          --match-id "${match}" --half "${half}" \
          --visual-cache "${VISUAL_CACHE}" \
          --visual-backbone "${VISUAL_BACKBONE}" \
          --window-size "${WINDOW_SIZE}" --stride "${STRIDE}" \
          --decode-threshold "${DECODE_THRESHOLD}" \
          --nms-mode "${NMS_MODE}" --nms-radius "${NMS_RADIUS}" \
          --device "${DEVICE}" \
          --out "${out}"; then
        echo "   (no ${half} for ${match} or inference failed; skipping)" >&2
        rm -f "${out}"
      fi
    done
  done

  # 2) Validate the schema before writing the zip.
  echo "--- validate ${variant} ---"
  python scripts/write_codabench_submission.py \
      --predictions "${preds_dir}" \
      --validate-only \
      --report "${report}"

  # 3) Package the submission zip.
  echo "--- package ${variant} ---"
  python scripts/write_codabench_submission.py \
      --predictions "${preds_dir}" \
      --out "${zip_out}"

  echo ">> wrote ${zip_out}"
done

echo
echo "Done. Per-model submissions:"
for variant in "${!CHECKPOINTS[@]}"; do
  echo "  ${OUT_ROOT}/${variant}/submission.zip"
done
