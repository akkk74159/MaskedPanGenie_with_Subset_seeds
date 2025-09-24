# ---------------- 核心共享庫 ----------------
add_library(PanGenieLib SHARED
    emissionprobabilitycomputer.cpp
    copynumber.cpp
    commandlineparser.cpp
    columnindexer.cpp
    dnasequence.cpp
    fastareader.cpp
    genotypingresult.cpp
    histogram.cpp
    hmm.cpp
    jellyfishcounter.cpp
    jellyfishreader.cpp
    kmerpath.cpp
    pathsampler.cpp
    probabilitycomputer.cpp
    probabilitytable.cpp
    sequenceutils.cpp
    timer.cpp
    transitionprobabilitycomputer.cpp
    threadpool.cpp
    uniquekmercomputer.cpp
    uniquekmers.cpp
    variant.cpp
    variantreader.cpp
    merstring.cpp)

# Include & flags
set_target_properties(PanGenieLib PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(PanGenieLib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(PanGenieLib PRIVATE -Wall -Wextra -O3 ${JELLYFISH_CFLAGS_OTHER})

target_link_libraries(PanGenieLib PUBLIC ${JELLYFISH_LIBRARIES})

# CUDA 依賴
if (USE_CUDA)
    target_link_libraries(PanGenieLib PUBLIC
        hmm_gpu                   # GPU kernels
        CUDA::cudart              # 動態 cudart.so
        cudadevrt                 # 靜態 dev runtime (.a)   
        cudart_static)            # 靜態 cudart (.a)
endif()

# ---------------- 執行檔 ----------------
add_executable(PanGenie pggtyper.cpp)
add_executable(PanGenie-graph pggtyper-graph.cpp)

foreach(bin PanGenie PanGenie-graph)
    target_link_libraries(${bin} PRIVATE PanGenieLib ${JELLYFISH_LDFLAGS_OTHER})
endforeach()
