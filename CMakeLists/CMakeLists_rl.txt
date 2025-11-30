cmake_minimum_required(VERSION 3.14)
project(rl_bindings LANGUAGES CXX)

# ===== Basics =====
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ===== Fail fast: required args (NO defaults, NO inference) =====
if(NOT DEFINED PY_MODULE_SRC)
  message(FATAL_ERROR "PY_MODULE_SRC must be defined (absolute path to bindings .cpp)")
endif()
if(NOT IS_ABSOLUTE "${PY_MODULE_SRC}")
  message(FATAL_ERROR "PY_MODULE_SRC must be absolute: ${PY_MODULE_SRC}")
endif()
if(NOT EXISTS "${PY_MODULE_SRC}")
  message(FATAL_ERROR "PY_MODULE_SRC not found: ${PY_MODULE_SRC}")
endif()

if(NOT DEFINED PROJ_ROOT)
  message(FATAL_ERROR "PROJ_ROOT must be defined (absolute project root)")
endif()
if(NOT IS_ABSOLUTE "${PROJ_ROOT}")
  message(FATAL_ERROR "PROJ_ROOT must be absolute: ${PROJ_ROOT}")
endif()
if(NOT IS_DIRECTORY "${PROJ_ROOT}")
  message(FATAL_ERROR "PROJ_ROOT is not a directory: ${PROJ_ROOT}")
endif()

# ===== Output root =====
if(NOT DEFINED PREFIX_DIR)
  set(PREFIX_DIR "${CMAKE_SOURCE_DIR}/dist")
endif()

# ===== Python & pybind11 =====
find_package(Python3 REQUIRED COMPONENTS Interpreter)
include(FetchContent)
set(PYBIND11_NEWPYTHON ON)
FetchContent_Declare(pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.12.0
)
FetchContent_MakeAvailable(pybind11)

# ===== Paths =====
set(CPP_INCLUDE_DIR "${PROJ_ROOT}/cpp/include")
set(CPP_SRC "${PROJ_ROOT}/cpp/src/rl.cpp")
if(NOT EXISTS "${CPP_SRC}")
  message(FATAL_ERROR "CPP source not found: ${CPP_SRC}")
endif()

# ===== Python package output =====
set(PY_PKG_NAME "rl")
set(PY_PKG_DIR "${PREFIX_DIR}/${PY_PKG_NAME}")
file(MAKE_DIRECTORY "${PY_PKG_DIR}")

# ===== Build =====
# rl_bindings.cpp(파이썬 바인딩) + rl.cpp(클래스 구현)를 함께 빌드해야 undefined symbol 방지
pybind11_add_module(rl MODULE "${PY_MODULE_SRC}" "${CPP_SRC}")
target_compile_features(rl PRIVATE cxx_std_17)
target_include_directories(rl PRIVATE
  "${CPP_INCLUDE_DIR}"
  "${PROJ_ROOT}/cpp/onnxruntime/include"
)

# 가독성 좋은 기본 경고(선택)
if(MSVC)
  target_compile_options(rl PRIVATE /W3)
else()
  target_compile_options(rl PRIVATE -Wall -Wextra -Wpedantic)
  # 심볼 최소화(선택) — pybind11가 필요한 심볼은 자동으로 export됨
  target_compile_options(rl PRIVATE -fvisibility=hidden)
endif()

set_target_properties(rl PROPERTIES
  PREFIX ""                                # 파이썬 모듈 접두사 비움 (import rl)
  LIBRARY_OUTPUT_DIRECTORY "${PY_PKG_DIR}" # .so/.pyd 출력 경로
  RUNTIME_OUTPUT_DIRECTORY "${PY_PKG_DIR}"
  OUTPUT_NAME "rl"                         # 모듈 파일명: rl.cpython-*.so
)

# RPATH (런타임 로더가 현재 폴더에서 찾도록)
if(APPLE)
  set_target_properties(rl PROPERTIES INSTALL_RPATH "@loader_path" BUILD_RPATH "@loader_path")
elseif(UNIX)
  set_target_properties(rl PROPERTIES INSTALL_RPATH "\$ORIGIN" BUILD_RPATH "\$ORIGIN")
endif()

# __init__.py 생성
file(GENERATE
  OUTPUT "${PY_PKG_DIR}/__init__.py"
  CONTENT "from .rl import *\n__all__ = ['RL']\n"
)

# ===== Info =====
message("RL INFO")
message(STATUS "PROJ_ROOT: ${PROJ_ROOT}")
message(STATUS "PY_MODULE_SRC: ${PY_MODULE_SRC}")
message(STATUS "PREFIX_DIR: ${PREFIX_DIR}")
message(STATUS "Python (interp): ${Python3_EXECUTABLE}")
message(STATUS "Output package directory: ${PY_PKG_DIR}")
message("")
