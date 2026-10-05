cmake_minimum_required(VERSION 3.24)
project(Ember VERSION 0.1.0 LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(EMBER_CUDA "Build the CUDA backend (needs the CUDA toolkit)" ON)
option(EMBER_WERROR "Treat compiler warnings as errors" ON)
option(EMBER_SANITIZE "Build with AddressSanitizer + UndefinedBehaviorSanitizer (GCC/Clang)" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(EMBER_WITH_CUDA OFF)
if(EMBER_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      # The GPU in this machine; pass -DCMAKE_CUDA_ARCHITECTURES=75;86;89 to build for others.
      set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(EMBER_WITH_CUDA ON)
    message(STATUS "Ember: CUDA backend enabled (CUDA ${CUDAToolkit_VERSION}, architectures ${CMAKE_CUDA_ARCHITECTURES})")
  else()
    message(STATUS "Ember: no CUDA compiler found; building the CPU backend only")
  endif()
endif()

# ---------------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------------
if(MSVC)
  set(EMBER_CXX_WARNINGS /W4 /permissive- /utf-8 /Zc:__cplusplus /wd4324)
  if(EMBER_WERROR)
    list(APPEND EMBER_CXX_WARNINGS /WX)
  endif()
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS WIN32_LEAN_AND_MEAN NOMINMAX)
else()
  set(EMBER_CXX_WARNINGS -Wall -Wextra -Wpedantic -Wshadow -Wnon-virtual-dtor -Wcast-align
                         -Wunused -Woverloaded-virtual -Wformat=2 -Wimplicit-fallthrough)
  if(EMBER_WERROR)
    list(APPEND EMBER_CXX_WARNINGS -Werror)
  endif()
  if(EMBER_SANITIZE)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address,undefined> $<$<COMPILE_LANGUAGE:CXX>:-fno-omit-frame-pointer>)
    add_link_options(-fsanitize=address,undefined)
  endif()
endif()

# ---------------------------------------------------------------------------
# The engine library
# ---------------------------------------------------------------------------
set(EMBER_SOURCES
  src/common/json.cpp
  src/common/log.cpp
  src/common/mmap_file.cpp
  src/common/thread_pool.cpp
  src/model/config.cpp
  src/model/safetensors.cpp
  src/model/unicode.cpp
  src/model/tokenizer.cpp
  src/model/chat_template.cpp
  src/backend/backend.cpp
  src/backend/sampling_host.cpp
  src/backend/cpu/cpu_kernels.cpp
  src/backend/cpu/cpu_kernels_avx2.cpp
  src/backend/cpu/cpu_backend.cpp
  src/kvcache/block_manager.cpp
  src/engine/engine.cpp
)

add_library(ember_core STATIC ${EMBER_SOURCES})
target_include_directories(ember_core PUBLIC src)
target_compile_options(ember_core PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${EMBER_CXX_WARNINGS}>)

# Only this file uses AVX2/FMA/F16C; the rest stays baseline x86-64 and picks
# the fast path at runtime.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64|i.86")
  if(MSVC)
    set_source_files_properties(src/backend/cpu/cpu_kernels_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
  else()
    set_source_files_properties(src/backend/cpu/cpu_kernels_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
  endif()
endif()

find_package(Threads REQUIRED)
target_link_libraries(ember_core PUBLIC Threads::Threads)
if(WIN32)
  target_link_libraries(ember_core PUBLIC ws2_32)
endif()

if(EMBER_WITH_CUDA)
  target_sources(ember_core PRIVATE
    src/backend/cuda/cuda_backend.cu
    src/backend/cuda/k_gemm.cu
    src/backend/cuda/k_elementwise.cu
    src/backend/cuda/k_attention.cu
    src/backend/cuda/k_sampling.cu
    src/backend/cuda/k_quant.cu
  )
  target_compile_definitions(ember_core PUBLIC EMBER_WITH_CUDA)
  target_link_libraries(ember_core PUBLIC CUDA::cudart_static)
  # (--expt-relaxed-constexpr is avoided: it triggers an nvcc 13.4 internal error with MSVC 14.5x.)
  set(EMBER_CUDA_FLAGS -lineinfo)
  if(MSVC)
    # C4211 comes from the stub files nvcc generates for kernels in anonymous namespaces.
    list(APPEND EMBER_CUDA_FLAGS -Xcompiler=/utf-8 -Xcompiler=/Zc:__cplusplus -Xcompiler=/wd4324 -Xcompiler=/wd4211)
  endif()
  target_compile_options(ember_core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${EMBER_CUDA_FLAGS}>)
endif()

# ---------------------------------------------------------------------------
# Programs and tests
# ---------------------------------------------------------------------------
add_executable(ember tools/ember_main.cpp src/server/server.cpp)
target_include_directories(ember PRIVATE tools)
target_link_libraries(ember PRIVATE ember_core)
target_compile_options(ember PRIVATE ${EMBER_CXX_WARNINGS})

enable_testing()
add_executable(ember_tests
  tests/test_main.cpp
  tests/test_common.cpp
  tests/test_tokenizer.cpp
  tests/test_model.cpp
  tests/test_cuda.cpp
  tests/test_engine.cpp
)
target_link_libraries(ember_tests PRIVATE ember_core)
target_include_directories(ember_tests PRIVATE tests)
target_compile_options(ember_tests PRIVATE ${EMBER_CXX_WARNINGS})
target_compile_definitions(ember_tests PRIVATE EMBER_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
add_test(NAME unit COMMAND ember_tests)
