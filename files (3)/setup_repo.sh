#!/usr/bin/env bash
# Assemble the InstanceDepth repository from:
#   $1 = directory containing the FLAT model files you uploaded
#        (dinov2_dpt.py, holistic_depth.py, matcher.py, ...)
#   $2 = the instancedepth_delivery directory (data engine + dataset + patches)
#   $3 = destination repo root (created)
#
# The uploaded files use relative imports (e.g. `from ..backbone.dinov2_dpt
# import ...`), so they MUST live in this exact package tree (it is the layout
# documented in the uploaded README.md).
set -euo pipefail
SRC="${1:?flat model files dir}"; DEL="${2:?delivery dir}"; DST="${3:?dest repo}"

P="$DST/instancedepth"
mkdir -p "$P"/{configs,models/{backbone,hdi,instance,refine},losses,utils,data,data_engine}

# --- top level -------------------------------------------------------------
cp "$SRC/build.py"                       "$P/"
cp "$SRC/registry.py"                    "$P/models/"
cp "$SRC/instance_depth.py"              "$P/models/"

# --- configs ----------------------------------------------------------------
cp "$SRC/run_config.py"                  "$P/configs/"
cp "$SRC/instance_depth.yaml"            "$P/configs/"
cp "$SRC/hdi_vitl.yaml"                  "$P/configs/"

# --- backbone / hdi ----------------------------------------------------------
cp "$SRC/dinov2_dpt.py"                  "$P/models/backbone/"
cp "$SRC/patch_attention.py"             "$P/models/hdi/"
cp "$SRC/range_decoder.py"               "$P/models/hdi/"
cp "$SRC/heads.py"                       "$P/models/hdi/"
cp "$SRC/holistic_depth.py"              "$P/models/hdi/"

# --- instance head (PATCHED matcher takes precedence) -------------------------
cp "$SRC/pixel_decoder.py"               "$P/models/instance/"
cp "$SRC/transformer_decoder.py"         "$P/models/instance/"
cp "$SRC/query_fusion.py"                "$P/models/instance/"
cp "$SRC/instance_head.py"               "$P/models/instance/"
cp "$DEL/patched/matcher.py"             "$P/models/instance/"   # GT-mask resize fix

# --- refinement ----------------------------------------------------------------
cp "$SRC/pair_selection.py"              "$P/models/refine/"
cp "$SRC/roi_extract.py"                 "$P/models/refine/"
cp "$SRC/relation_reason.py"             "$P/models/refine/"
cp "$SRC/occlusion_refine.py"            "$P/models/refine/"

# --- losses / utils -------------------------------------------------------------
cp "$SRC/hdi_losses.py"                  "$P/losses/"
cp "$SRC/instance_losses.py"             "$P/losses/"
cp "$SRC/refine_losses.py"               "$P/losses/"
cp "$SRC/checkpoint.py"                  "$P/utils/"
cp "$SRC/shapes.py"                      "$P/utils/"

# --- data engine + dataset (this delivery) ---------------------------------------
cp -r "$DEL/instancedepth/data_engine/." "$P/data_engine/"
cp -r "$DEL/instancedepth/data/."        "$P/data/"
cp    "$DEL/instancedepth/configs/gid_custom.yaml" "$P/configs/"
cp    "$DEL/train.py" "$DEL/evaluate.py" "$DST/" 2>/dev/null || true

# --- package markers ---------------------------------------------------------------
for d in "$P" "$P/configs" "$P/models" "$P/models/backbone" "$P/models/hdi" \
         "$P/models/instance" "$P/models/refine" "$P/losses" "$P/utils" \
         "$P/data" "$P/data_engine"; do
  touch "$d/__init__.py"
done
find "$P" -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
echo "Repository assembled at: $DST"
find "$DST" -name "*.py" | wc -l
