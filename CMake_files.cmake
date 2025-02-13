include_directories(${CMAKE_CURRENT_SOURCE_DIR}/includes
        ${Protobuf_INCLUDE_DIRS}
        ${ONNX_INCLUDE_DIRS}
        ${GLOG_INCLUDE_DIRS}
        ${OpenMP_INCLUDE}
        ${ARMADILLO_INCLUDE_DIRS}
)

link_directories(${GLOG_LIBRARY_DIRS}
        ${ONNX_LIBRARIES}
)

# 匹配指定目录下的所有 .cpp 文件（包括子目录）
file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/source/data/core/*.cpp)
file(GLOB_RECURSE RUNTIME_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/source/runtime/*.cpp)
file(GLOB_RECURSE CPU_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/source/cpu/*.cpp)
file(GLOB_RECURSE COMMON_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/source/common/*.cpp)
file(GLOB_RECURSE C_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/source/c/*.cpp)

add_executable(BatmanInfer main.cpp
        test/tensor_get_values.cpp
        ${SOURCES}
        ${RUNTIME_SOURCES}
        ${CPU_SOURCES}
        ${C_SOURCES}
        ${COMMON_SOURCES}
        test/test_layer_operator.cpp
        test/test_neon_operator.cpp
        #        test/test_onnx.cpp
        #        source/others/utils.cpp
        #        source/onnx_conv/OnnxUtils.cpp
        #        source/ir.cpp
        #        source/runtime_attr.cpp
        #        source/runtime_op.cpp
        #        source/runtime_ir.cpp
        #        source/tensor_util.cpp
        #        test/test_topo.cpp
        #        source/operators/batman_operator.cpp
        #        source/layer/abstract/layer.cpp
        #        source/layer/abstract/layer_factory.cpp
        #        source/layer/abstract/param_layer.cpp
        #        source/layer/detail/relu.cpp
        #        source/layer/detail/softmax.cpp
        #        source/layer/detail/sigmoid.cpp
        #        source/layer/detail/maxpooling.cpp
        #        test/test_halide_drom.cpp
        #        source/layer/detail/convolution.cpp
        #        test/test_convolution.cpp
        #        test/test_resnt.cpp
        #        source/runtime/bi_memory.cpp
)

# 链接GoogleTest库
target_link_libraries(BatmanInfer
        ${ONNX_LIBRARIES}/libonnx_proto.a
        GTest::gtest_main
        ${ARMADILLO_LIBRARIES}
        ${GLOG_LIBRARIES}
        ${Protobuf_LIBRARIES}
        ${OpenCV_LIBS}
        ${ONNX_LIBRARIES}/libonnx.a
        ${OpenMP_omp_LIBRARY}
        ${ABSEL_LIBARARY}
        ${UTF8RANGE_LIBRARY}
)

if (ANDROID)
    target_link_libraries(BatmanInfer OpenMP::OpenMP_CXX)
endif ()

# 添加测试
include(GoogleTest)
#gtest_discover_tests(BatmanInfer)

# 包含Armadillo头文件路径
target_include_directories(BatmanInfer PRIVATE ${ARMADILLO_INCLUDE_DIRS})
