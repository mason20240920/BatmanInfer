# 匹配指定目录下的所有 .cpp 文件（包括子目录）
file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/source/data/core/*.cpp)
file(GLOB_RECURSE RUNTIME_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/source/runtime/*.cpp)
file(GLOB_RECURSE CPU_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/source/cpu/*.cpp)
file(GLOB_RECURSE COMMON_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/source/common/*.cpp)
file(GLOB_RECURSE C_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/source/c/*.cpp)
if (ENABLE_BENCHMARK)
    file(GLOB_RECURSE BENCHMARK_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/benchmark/*.cpp)
else ()
    file(GLOB_RECURSE BENCHMARK_SOURCES "")
endif ()

add_executable(BatmanInfer main.cpp
        test/tensor_get_values.cpp
        ${SOURCES}
        ${RUNTIME_SOURCES}
        ${CPU_SOURCES}
        ${C_SOURCES}
        ${COMMON_SOURCES}
        test/test_layer_operator.cpp
        test/test_neon_operator.cpp
        test/test_dynamic_gemmlowp.cpp
        test/test_quantize.cpp
        test/test_mem_alloc.cpp
        source/utils/utils.cpp
        test/test_matmul_assem.cpp
        test/test_perf_model.cpp
        ${BENCHMARK_SOURCES}
        test/test_kvcaches.cpp
        test/gemm_lowp_outputstage_test.cpp
)

if (ENABLE_BENCHMARK)
    add_executable(matmul_benchmark      # 基准测试程序
            benchmark/benchmark_test/matmul_op_test.cpp
            ${SOURCES}
            ${RUNTIME_SOURCES}
            ${CPU_SOURCES}
            ${C_SOURCES}
            ${COMMON_SOURCES}
            source/utils/utils.cpp
    )
endif ()
