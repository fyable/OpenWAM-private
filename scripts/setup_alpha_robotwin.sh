#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
BASE_PYTHON="${BASE_PYTHON:-${ROOT}/.venv-libero/bin/python}"
SOURCE_PACKAGES="${ROBOTWIN_SOURCE_PACKAGES:-/data2/zhibo/LHN/wam/envs/robotwin/lib/python3.10/site-packages}"
SOURCE_REPO="${ROBOTWIN_SOURCE_REPO:-/data1/cuihanyun/RoboTwin}"
SOURCE_CUROBO="${ROBOTWIN_SOURCE_CUROBO:-/data2/zhibo/LHN/wam/robotwin/RoboTwin/envs/curobo}"
SOURCE_ASSETS="${ROBOTWIN_ASSETS:-/data2/vla-data/robotwin/assets}"
ENV_ROOT="${ROOT}/.venv-robotwin"
LOCAL_REPO="${ROOT}/third_party/RoboTwin"

copy_tree() {
    mkdir -p "$2"
    tar --exclude=__pycache__ --exclude=.git --exclude='gitdb*' --exclude='smmap*' -C "$1" -cf - . | \
        tar --no-same-owner --no-same-permissions -C "$2" -xf -
}

for required in "${SOURCE_PACKAGES}/sapien" "${SOURCE_PACKAGES}/torch" \
    "${SOURCE_REPO}/script/eval_policy.py" "${SOURCE_CUROBO}/src/curobo" \
    "${SOURCE_ASSETS}/embodiments/aloha-agilex" "${SOURCE_ASSETS}/objects"; do
    [[ -e "${required}" ]] || { echo "Missing offline source: ${required}" >&2; exit 1; }
done

if [[ ! -f "${ENV_ROOT}/.robotwin-copy-complete" ]]; then
    [[ -x "${ENV_ROOT}/bin/python" ]] || "${BASE_PYTHON}" -m venv "${ENV_ROOT}"
    echo "Copying simulator dependencies into ${ENV_ROOT} (about 9 GB)..."
    copy_tree "${SOURCE_PACKAGES}" "${ENV_ROOT}/lib/python3.10/site-packages"
    touch "${ENV_ROOT}/.robotwin-copy-complete"
fi

if [[ ! -f "${LOCAL_REPO}/.openwam-copy-complete" ]]; then
    mkdir -p "${LOCAL_REPO}/envs"
    for name in script task_config description policy code_gen; do
        copy_tree "${SOURCE_REPO}/${name}" "${LOCAL_REPO}/${name}"
    done
    for source in "${SOURCE_REPO}"/envs/*; do
        [[ "$(basename "${source}")" == curobo || "$(basename "${source}")" == __pycache__ ]] && continue
        if [[ -d "${source}" ]]; then
            copy_tree "${source}" "${LOCAL_REPO}/envs/$(basename "${source}")"
        else
            cp "${source}" "${LOCAL_REPO}/envs/"
        fi
    done
    copy_tree "${SOURCE_CUROBO}" "${LOCAL_REPO}/envs/curobo"
    [[ -e "${LOCAL_REPO}/assets" ]] || ln -s "${SOURCE_ASSETS}" "${LOCAL_REPO}/assets"
    mkdir -p "${LOCAL_REPO}/eval_result"
    touch "${LOCAL_REPO}/.openwam-copy-complete"
fi

if ! "${ENV_ROOT}/bin/python" -c 'import gitdb, smmap' 2>/dev/null; then
    "${ENV_ROOT}/bin/python" -m pip install --no-index --no-deps --ignore-installed \
        --find-links "${ROOT}/.cache/robotwin-wheels" gitdb==4.0.12 smmap==5.0.3
fi
"${ENV_ROOT}/bin/python" scripts/configure_robotwin_local.py
"${ENV_ROOT}/bin/python" scripts/check_robotwin_env.py
echo "Environment ready. Source: ${ROOT}/outputs/robotwin/environment.sh"
