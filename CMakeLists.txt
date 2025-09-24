add_compile_options(-mlong-double-64)
cmake_minimum_required(VERSION 3.18)

# ***** 關鍵：使用動態 cudart，並手動告訴 linker 到 /usr/local/cuda/lib64 *****
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

project(PanGenie LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
option(USE_CUDA "Enable CUDA HMM kernels" ON)

# ---------------- CUDA Toolkit ----------------
if (USE_CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

# ---------------- Jellyfish ----------------
find_package(PkgConfig REQUIRED)
pkg_check_modules(JELLYFISH REQUIRED jellyfish-2.0)
include_directories(${JELLYFISH_INCLUDE_DIRS})
link_directories(${JELLYFISH_LIBRARY_DIRS})

# ***** Method A：將 CUDA 靜態庫所在目錄加入全域 linker path *****
if (USE_CUDA)
    link_directories(/usr/local/cuda/lib64)  # 若 toolkit 安裝於其他路徑請修改
endif()

# ---------------- GPU 靜態庫 (hmm_gpu) ----------------
if (USE_CUDA)
    add_library(hmm_gpu STATIC src/hmm_gpu.cu)
    set_target_properties(hmm_gpu PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_include_directories(hmm_gpu PUBLIC src)
    target_link_libraries(hmm_gpu PUBLIC CUDA::cudart)
    target_link_options(hmm_gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--cudart=shared>)
endif()

# 讓所有 target 都能看到 USE_CUDA
if (USE_CUDA)
    add_compile_definitions(USE_CUDA)
endif()

# ---------------- 進入 src/ 與 tests/ ----------------
add_subdirectory(src)

#enable_testing()
#add_subdirectory(tests)
#add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} DEPENDS tests)
