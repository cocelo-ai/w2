#!/usr/bin/env bash
set -euo pipefail

# =======================================
# build.sh
# - Assumes CMakeLists_*.txt are in: ${PROJECT_ROOT}/CMakeLists/
# - Components: mode, rl, robot, onnxpolicy
# - Build trees & outputs: ${PROJECT_ROOT}/sdk/classes/<component>
# - Options: --clean (clean then rebuild), --clean-only (clean and exit)
# =======================================
#
# Usage:
#   bash build.sh                  # incremental build (Release by default)
#   BUILD_TYPE=Debug bash build.sh # change build type
#   bash build.sh --clean          # clean then rebuild
#   bash build.sh --clean-only     # clean and exit
#
# Extra cmake args can be provided via env var:
#   CMAKE_EXTRA_ARGS="-DUSE_SOMETHING=ON" bash build.sh
# =======================================

# Guard against "bash build.sh" accidentally run as "bash" without script path
SCRIPT_PATH="$0"
if [[ "$SCRIPT_PATH" == "bash" || "$SCRIPT_PATH" == "-bash" ]]; then
  echo "Please run: bash build.sh"
  exit 1
fi

PROJECT_ROOT="$(cd -- "$(dirname -- "$SCRIPT_PATH")" && pwd -P)"
CMAKE_LISTS_DIR="${PROJECT_ROOT}/CMakeLists"
BUILD_BASE="${PROJECT_ROOT}/sdk/classes"
BUILD_TYPE="${BUILD_TYPE:-Debug}"  # Release or Debug
CMAKE_EXTRA_ARGS="${CMAKE_EXTRA_ARGS:-}"

mkdir -p "${BUILD_BASE}"

# ---- Python detection (conda > python3 > python) ----
if command -v conda >/dev/null 2>&1 && conda info --envs >/dev/null 2>&1; then
  if command -v python >/dev/null 2>&1; then
    PYTHON="$(command -v python)"
  else
    PYTHON="$(command -v python3 || true)"
  fi
else
  PYTHON="$(command -v python3 || command -v python)"
fi
if [[ -z "${PYTHON:-}" ]]; then
  echo "ERROR: Python interpreter not found."
  exit 2
fi

# ---- Options ----
CLEAN=false
CLEAN_ONLY=false
for arg in "$@"; do
  case "$arg" in
    --clean) CLEAN=true ;;
    --clean-only) CLEAN=true; CLEAN_ONLY=true ;;
  esac
done

if $CLEAN; then
  echo "[CLEAN] Removing ${BUILD_BASE}/{mode,rl,robot,onnxpolicy,joystick} ..."
  rm -rf "${BUILD_BASE}/mode" "${BUILD_BASE}/rl" "${BUILD_BASE}/robot" "${BUILD_BASE}/onnxpolicy" "${BUILD_BASE}/joystick"
  echo "[CLEAN] Done."
  if $CLEAN_ONLY; then
    exit 0
  fi
fi

# ---- Helpers ----
# install_cmakelists <component> <filename>
install_cmakelists() {
  local comp="$1"; shift
  local fname="$1"; shift || true

  local src="${CMAKE_LISTS_DIR}/${fname}"
  if [[ ! -f "${src}" ]]; then
    echo "ERROR: ${src} not found. Expected under ${CMAKE_LISTS_DIR}."
    exit 3
  fi
  install -D "${src}" "${BUILD_BASE}/${comp}/CMakeLists.txt"
}

# configure_build <component> <py_module_src> <extra_cmake_args...>
configure_build() {
  local comp="$1"; shift
  local py_src="$1"; shift
  local extra_args=("$@")

  cmake -S "${BUILD_BASE}/${comp}" -B "${BUILD_BASE}/${comp}" -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DPREFIX_DIR="${BUILD_BASE}" \
    -DPython3_EXECUTABLE="${PYTHON}" \
    -DPROJ_ROOT="${PROJECT_ROOT}" \
    -DPY_MODULE_SRC="${py_src}" \
    ${extra_args[@]+"${extra_args[@]}"} \
    ${CMAKE_EXTRA_ARGS}

  cmake --build "${BUILD_BASE}/${comp}" --config "${BUILD_TYPE}" -j
}

# Common sources/dirs
RL_SRC="${PROJECT_ROOT}/cpp/src/rl_bindings.cpp"
MODE_SRC="${PROJECT_ROOT}/cpp/src/mode_bindings.cpp"
ROBOT_SRC="${PROJECT_ROOT}/cpp/src/robot_bindings.cpp"
ONNXPOLICY_SRC="${PROJECT_ROOT}/cpp/src/onnxpolicy_bindings.cpp"
ONNXRUNTIME_DIR="${PROJECT_ROOT}/cpp/onnxruntime"
JOYSTICK_SRC="${PROJECT_ROOT}/cpp/src/joystick_bindings.cpp"

# -----------------------------------------------------------------------------
# If an ONNX Runtime distribution has not been provisioned yet, fetch it.
#
# The project expects a pre-built ONNX Runtime C API to be available under
# ${PROJECT_ROOT}/cpp/onnxruntime with `include` and `lib` subdirectories.  When
# building for the first time on a new machine, this directory may be absent.
# If so, determine the host architecture, download the appropriate v1.23.0
# release from the official ONNX Runtime GitHub, extract it into place, and
# ensure `include` and `lib` exist.  This block is intentionally placed here to
# run before any CMake configuration occurs.
if [[ ! -d "${ONNXRUNTIME_DIR}" || ! -d "${ONNXRUNTIME_DIR}/include" || ! -d "${ONNXRUNTIME_DIR}/lib" ]]; then
  echo "[INFO] ONNX Runtime not found at ${ONNXRUNTIME_DIR}. Downloading pre-built v1.23.0 package..."
  # Determine architecture (translate uname -m to ONNX Runtime asset suffix).
  ARCH="$(uname -m)"
  case "${ARCH}" in
    x86_64)
      ORT_ARCH="linux-x64"
      ;;
    aarch64|arm64)
      ORT_ARCH="linux-aarch64"
      ;;
    *)
      echo "ERROR: Unsupported architecture: ${ARCH}" >&2
      exit 4
      ;;
  esac
  ORT_VERSION="1.23.0"
  FILENAME="onnxruntime-${ORT_ARCH}-${ORT_VERSION}.tgz"
  URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${FILENAME}"
  TEMP_DIR="$(mktemp -d)"
  echo "[INFO] Downloading ${URL} ..."
  # Prefer wget; fall back to curl if wget is unavailable.
  if command -v wget >/dev/null 2>&1; then
    wget -q -O "${TEMP_DIR}/${FILENAME}" "${URL}"
  elif command -v curl >/dev/null 2>&1; then
    curl -L -o "${TEMP_DIR}/${FILENAME}" "${URL}"
  else
    echo "ERROR: Neither wget nor curl is available to download ${FILENAME}" >&2
    exit 4
  fi
  echo "[INFO] Extracting ${FILENAME} ..."
  tar -xf "${TEMP_DIR}/${FILENAME}" -C "${TEMP_DIR}"
  # Locate the extracted directory; different tarballs may unpack into a
  # directory whose name encodes architecture and version.  Find it by
  # pattern rather than relying on a fixed name.
  EXTRACT_DIR=""
  for d in "${TEMP_DIR}"/onnxruntime-*-${ORT_VERSION}; do
    if [[ -d "$d" ]]; then
      EXTRACT_DIR="$d"
      break
    fi
  done
  if [[ -z "${EXTRACT_DIR}" ]]; then
    # Fallback: assume the contents were extracted directly into TEMP_DIR
    EXTRACT_DIR="${TEMP_DIR}"
  fi
  mkdir -p "${ONNXRUNTIME_DIR}"
  # Copy all files from the extracted package into the target directory.  Using
  # `.` ensures hidden files (if any) are copied.
  cp -R "${EXTRACT_DIR}/." "${ONNXRUNTIME_DIR}/"
  # Some packages provide libraries under lib64; unify under lib for this project.
  if [[ ! -d "${ONNXRUNTIME_DIR}/lib" && -d "${ONNXRUNTIME_DIR}/lib64" ]]; then
    mv "${ONNXRUNTIME_DIR}/lib64" "${ONNXRUNTIME_DIR}/lib"
  fi
  # Clean up temporary directory.
  rm -rf "${TEMP_DIR}"
  # Validate the installation.
  if [[ ! -d "${ONNXRUNTIME_DIR}/include" || ! -d "${ONNXRUNTIME_DIR}/lib" ]]; then
    echo "ERROR: Failed to set up ONNX Runtime. Missing include or lib directories." >&2
    exit 5
  fi
  echo "[INFO] ONNX Runtime installed under ${ONNXRUNTIME_DIR}"
fi

# ===== onnxpolicy =====
install_cmakelists "onnxpolicy" "CMakeLists_onnxpolicy.txt"
configure_build "onnxpolicy" "${ONNXPOLICY_SRC}" -DONNXRUNTIME_DIR="${ONNXRUNTIME_DIR}"

# ===== mode =====
install_cmakelists "mode" "CMakeLists_mode.txt"
configure_build "mode" "${MODE_SRC}"

# ===== rl =====
install_cmakelists "rl" "CMakeLists_rl.txt"
configure_build "rl" "${RL_SRC}"

# ===== robot =====
install_cmakelists "robot" "CMakeLists_robot.txt"
configure_build "robot" "${ROBOT_SRC}"

# ===== joystick =====
install_cmakelists "joystick" "CMakeLists_joystick.txt"
configure_build "joystick" "${JOYSTICK_SRC}"

echo "Done! Build output is under: ${BUILD_BASE}"
