cmake_minimum_required(VERSION 3.14)
project(mode_bindings LANGUAGES CXX)

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

# ===== Python package output =====
set(PY_PKG_DIR "${PREFIX_DIR}/mode")
file(MAKE_DIRECTORY "${PY_PKG_DIR}")

# ===== Sources (구현 포함) =====
set(CPP_SOURCES
  "${PROJ_ROOT}/cpp/src/mode.cpp"          # ADDED
  "${PROJ_ROOT}/cpp/src/onnxpolicy.cpp"    # ADDED: onnxpolicy 구현을 같이 빌드 (undefined symbol 해결)
)

# ===== Build =====
pybind11_add_module(mode MODULE "${PY_MODULE_SRC}" ${CPP_SOURCES})  # CHANGED: 구현 소스 추가
target_compile_features(mode PRIVATE cxx_std_17)
target_include_directories(mode PRIVATE "${CPP_INCLUDE_DIR}")

# ===== onnxruntime (옵션: 외부에서 경로 주면 사용, 없으면 w2/cpp/onnxruntime 힌트) =====
# - 모듈이 직접 ORT 심볼을 쓰지 않더라도, 런타임 편의를 위해 발견되면 include/link/copy를 수행
# - 외부에서 ORT_INC/ORT_LIBDIR/ONNXRUNTIME_LIB 를 이미 넘겼다면 그대로 사용
# - 아니면 ONNXRUNTIME_DIR 또는 기본 힌트(${PROJ_ROOT}/cpp/onnxruntime)를 참고
if(NOT DEFINED ORT_INC OR NOT DEFINED ORT_LIBDIR)                                     # ADDED
  if(DEFINED ONNXRUNTIME_DIR)                                                         # ADDED
    set(_ORT_ROOT "${ONNXRUNTIME_DIR}")                                               # ADDED
  else()                                                                              # ADDED
    set(_ORT_ROOT "${PROJ_ROOT}/cpp/onnxruntime")                                     # ADDED
  endif()                                                                              # ADDED
  if(NOT DEFINED ORT_INC)                                                             # ADDED
    set(ORT_INC "${_ORT_ROOT}/include")                                               # ADDED
  endif()                                                                             # ADDED
  if(NOT DEFINED ORT_LIBDIR)                                                          # ADDED
    set(ORT_LIBDIR "${_ORT_ROOT}/lib")                                                # ADDED
  endif()                                                                             # ADDED
endif()                                                                                # ADDED

if(EXISTS "${ORT_INC}")
  target_include_directories(mode PRIVATE "${ORT_INC}")                                # ADDED: ORT 헤더 발견 시 포함
endif()

# 있으면 링크 (필수 아님)
find_library(ONNXRUNTIME_LIB                                                           # ADDED
  NAMES onnxruntime libonnxruntime
  HINTS "${ORT_LIBDIR}"
)
if(ONNXRUNTIME_LIB)
  target_link_libraries(mode PRIVATE "${ONNXRUNTIME_LIB}")                             # ADDED
endif()

set_target_properties(mode PROPERTIES
  PREFIX ""
  LIBRARY_OUTPUT_DIRECTORY "${PY_PKG_DIR}"
  RUNTIME_OUTPUT_DIRECTORY "${PY_PKG_DIR}"
  OUTPUT_NAME "mode"
)

# RPATH
if(APPLE)
  set_target_properties(mode PROPERTIES INSTALL_RPATH "@loader_path" BUILD_RPATH "@loader_path")
elseif(UNIX)
  set_target_properties(mode PROPERTIES INSTALL_RPATH "\$ORIGIN" BUILD_RPATH "\$ORIGIN")
endif()

# __init__.py
file(GENERATE
  OUTPUT "${PY_PKG_DIR}/__init__.py"
  CONTENT "from .mode import *\n__all__ = ['Mode']\n\n"
)

# onnxruntime 공유 라이브러리를 모듈 옆에 복사 (발견된 경우만)
set(_ORT_DLL_CAND "")                                                                   # ADDED
if(DEFINED ORT_LIBDIR AND EXISTS "${ORT_LIBDIR}")                                       # ADDED
  file(GLOB _cand_so    "${ORT_LIBDIR}/libonnxruntime.so*")                             # ADDED
  file(GLOB _cand_dylib "${ORT_LIBDIR}/libonnxruntime*.dylib")                          # ADDED
  file(GLOB _cand_dll   "${ORT_LIBDIR}/onnxruntime*.dll" "${ORT_LIBDIR}/onnxruntime.dll") # ADDED
  if(_cand_so)
    list(GET _cand_so 0 _ORT_DLL_CAND)
  elseif(_cand_dylib)
    list(GET _cand_dylib 0 _ORT_DLL_CAND)
  elseif(_cand_dll)
    list(GET _cand_dll 0 _ORT_DLL_CAND)
  endif()
endif()

if(EXISTS "${_ORT_DLL_CAND}")                                                            # ADDED
  add_custom_command(TARGET mode POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_ORT_DLL_CAND}" "${PY_PKG_DIR}"
  )
endif()

message("MODE INFO")
message(STATUS "PROJ_ROOT: ${PROJ_ROOT}")
message(STATUS "PY_MODULE_SRC: ${PY_MODULE_SRC}")
message(STATUS "ONNXRUNTIME_DIR: ${ONNXRUNTIME_DIR}")
message(STATUS "ORT_INC: ${ORT_INC}")
message(STATUS "ORT_LIBDIR: ${ORT_LIBDIR}")
message(STATUS "ONNXRUNTIME_LIB: ${ONNXRUNTIME_LIB}")
message(STATUS "PREFIX_DIR: ${PREFIX_DIR}")
message(STATUS "Python (interp): ${Python3_EXECUTABLE}")
message(STATUS "Output package directory: ${PY_PKG_DIR}")
message("")
