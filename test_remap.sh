#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${REPO_ROOT}"

PYTHON_BIN=${PYTHON_BIN:-python}
DEVICE=${DEVICE:-0}
DATA_BASE=${DATA_BASE:-}
CACHE_ROOT=${CACHE_ROOT:-${REPO_ROOT}/cache/remap/main}
RESULT_ROOT=${RESULT_ROOT:-${REPO_ROOT}/results/remap/main}
BATCH_SIZE=${BATCH_SIZE:-4}
WORKERS=${WORKERS:-8}
FORCE=${FORCE:-0}
DRY_RUN=${DRY_RUN:-0}

MVTEC_CHECKPOINT=${MVTEC_CHECKPOINT:-${REPO_ROOT}/checkpoints/9_12_4_multiscale/epoch_15.pth}
VISA_CHECKPOINT=${VISA_CHECKPOINT:-${REPO_ROOT}/checkpoints/9_12_4_multiscale_visa/epoch_15.pth}

print_command() {
  printf 'COMMAND:'
  printf ' %q' "$@"
  printf '\n'
}

execute() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    print_command "$@"
  else
    CUDA_VISIBLE_DEVICES=${DEVICE} ANOMALYCLIP_FAST_AUPRO=1 "$@"
  fi
}

complete_set() {
  local directory=$1
  shift
  local filename
  for filename in "$@"; do
    [[ -f "${directory}/${filename}" ]] || return 1
  done
}

run_one() {
  local slug=$1 dataset=$2 data_path=$3 checkpoint=$4 class_bank=$5
  local root=${CACHE_ROOT}/${slug}
  local base=${root}/base
  local rate_scores=${root}/rate_scores.npy
  local crop=${root}/crop
  local work=${root}/work_maps
  local output=${RESULT_ROOT}/${slug}.json
  local semantic=crop_semantic_scores_${class_bank}.npy

  echo "== ReMAP: ${slug} =="
  if [[ "${DRY_RUN}" != "1" ]]; then
    [[ -d "${data_path}" ]] || { echo "dataset not found: ${data_path}" >&2; exit 1; }
    [[ -f "${data_path}/meta.json" ]] || { echo "metadata not found: ${data_path}/meta.json" >&2; exit 1; }
    [[ -f "${checkpoint}" ]] || { echo "checkpoint not found: ${checkpoint}" >&2; exit 1; }
  fi

  if [[ "${FORCE}" == "1" ]] || ! complete_set "${base}" \
      metadata.json official_scores.npy masks.npy identity_features.npy \
      text_bank_scores.npy; then
    execute "${PYTHON_BIN}" -m rate_prompt.cache_dataset \
      --data_path "${data_path}" --dataset "${dataset}" \
      --checkpoint_path "${checkpoint}" --output "${base}" --identity_only \
      --image_size 518 --depth 9 --n_ctx 12 --t_n_ctx 4 \
      --batch_size "${BATCH_SIZE}" --workers "${WORKERS}"
  fi

  if [[ "${FORCE}" == "1" || ! -f "${rate_scores}" ]]; then
    execute "${PYTHON_BIN}" -m rate_prompt.cache_pyramid_rate \
      --cache "${base}" --checkpoint_path "${checkpoint}" \
      --output "${rate_scores}" --auxiliary_side 16 --interpolation bicubic \
      --image_size 518 --depth 9 --n_ctx 12 --t_n_ctx 4
  fi

  if [[ "${FORCE}" == "1" ]] || ! complete_set "${crop}" \
      metadata.json crop_scores.npy crop_features.npy geometry.npy \
      crop_semantic_scores.npy; then
    execute "${PYTHON_BIN}" -m rate_fovea.cache \
      --data_path "${data_path}" --dataset "${dataset}" \
      --checkpoint_path "${checkpoint}" --branch_cache "${base}" \
      --guidance_scores "${rate_scores}" --semantic_cache "${base}" \
      --semantic_bank route_state --output "${crop}" \
      --crop_size 252 --sigma_span 2 --image_size 518 --final_layer_only \
      --depth 9 --n_ctx 12 --t_n_ctx 4 \
      --batch_size "${BATCH_SIZE}" --workers "${WORKERS}"
  fi

  if [[ "${FORCE}" == "1" || ! -f "${crop}/${semantic}" ]]; then
    execute "${PYTHON_BIN}" -m rate_fovea.cache_crop_semantic \
      --fovea_cache "${crop}" --checkpoint_path "${checkpoint}" \
      --bank "${class_bank}" --image_size 518 \
      --depth 9 --n_ctx 12 --t_n_ctx 4 --batch_size 16
  fi

  if [[ "${FORCE}" == "1" || ! -f "${output}" ]]; then
    execute "${PYTHON_BIN}" -m rate_fovea.evaluate_prompt_cache \
      --cache "${base}" --fovea_cache "${crop}" \
      --rate_scores_file "${rate_scores}" --crop_semantic_file "${semantic}" \
      --identity_features "${base}/identity_features.npy" \
      --feature_steps 4 --full_feature_steps 1 --full_graph_scope fovea \
      --full_semantic_persistence --direct_intermediate \
      --final_only --promoted_only --gaussian_backend torch --method remap \
      --batch_size "${BATCH_SIZE}" --workers "${WORKERS}" \
      --work_dir "${work}" --output "${output}"
  fi
}

resolve_data_path() {
  local variable_name=$1
  local relative_path=$2
  local explicit_path=${!variable_name:-}
  if [[ -n "${explicit_path}" ]]; then
    printf '%s\n' "${explicit_path}"
  elif [[ -n "${DATA_BASE}" ]]; then
    printf '%s\n' "${DATA_BASE}/${relative_path}"
  else
    echo "set DATA_BASE or ${variable_name} before running ReMAP" >&2
    return 2
  fi
}

run_named() {
  case "$1" in
    mvtec) run_one mvtec mvtec "$(resolve_data_path MVTEC_ROOT mvdataset)" "${MVTEC_CHECKPOINT}" structural ;;
    visa) run_one visa visa "$(resolve_data_path VISA_ROOT Visa)" "${VISA_CHECKPOINT}" structural ;;
    btad) run_one btad btad "$(resolve_data_path BTAD_ROOT BTech_Dataset_transformed)" "${MVTEC_CHECKPOINT}" structural ;;
    mpdd) run_one mpdd mpdd "$(resolve_data_path MPDD_ROOT mpdd)" "${MVTEC_CHECKPOINT}" structural ;;
    sdd) run_one sdd sdd "$(resolve_data_path SDD_ROOT SDD)" "${MVTEC_CHECKPOINT}" structural ;;
    dagm) run_one dagm dagm "$(resolve_data_path DAGM_ROOT DAGM_KaggleUpload)" "${MVTEC_CHECKPOINT}" structural ;;
    dtd) run_one dtd dtd-synthetic "$(resolve_data_path DTD_ROOT DTD-Synthetic)" "${MVTEC_CHECKPOINT}" structural ;;
    isic) run_one isic isic "$(resolve_data_path ISIC_ROOT ISBI)" "${MVTEC_CHECKPOINT}" clinical ;;
    colondb) run_one colondb cvc-colondb "$(resolve_data_path COLONDB_ROOT medical/CVC-ColonDB)" "${MVTEC_CHECKPOINT}" clinical ;;
    clinicdb) run_one clinicdb cvc-clinicdb "$(resolve_data_path CLINICDB_ROOT medical/CVC-ClinicDB)" "${MVTEC_CHECKPOINT}" clinical ;;
    kvasir) run_one kvasir kvasir "$(resolve_data_path KVASIR_ROOT medical/Kvasir)" "${MVTEC_CHECKPOINT}" clinical ;;
    endo) run_one endo endo "$(resolve_data_path ENDO_ROOT medical/EndoTect_2020_Segmentation_Test_Dataset)" "${MVTEC_CHECKPOINT}" clinical ;;
    tn3k) run_one tn3k tn3k "$(resolve_data_path TN3K_ROOT Thyroid_Dataset/tn3k)" "${MVTEC_CHECKPOINT}" clinical ;;
    *) echo "unknown dataset '$1'" >&2; exit 2 ;;
  esac
}

run_group() {
  local dataset
  for dataset in "$@"; do run_named "${dataset}"; done
}

case "${1:-}" in
  ""|-h|--help)
    echo "usage: DATA_BASE=/workspace/data/path $0 [all|industrial|medical|DATASET]"
    exit 0
    ;;
  all) run_group mvtec visa btad mpdd sdd dtd isic colondb clinicdb kvasir endo tn3k dagm ;;
  industrial) run_group mvtec visa btad mpdd sdd dtd dagm ;;
  medical) run_group isic colondb clinicdb kvasir endo tn3k ;;
  mvtec|visa|btad|mpdd|sdd|dagm|dtd|isic|colondb|clinicdb|kvasir|endo|tn3k) run_named "$1" ;;
  *) echo "usage: DATA_BASE=/workspace/data/path $0 [all|industrial|medical|DATASET]" >&2; exit 2 ;;
esac

if [[ "${DRY_RUN}" != "1" ]]; then
  "${PYTHON_BIN}" -m remap.report_results \
    --results_root "${RESULT_ROOT}" --output "${RESULT_ROOT}/summary.json"
fi
