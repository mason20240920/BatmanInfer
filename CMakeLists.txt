cmake_minimum_required(VERSION 3.28)
project(BatmanInfer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)

# 检查目标平台
if(ANDROID)
    # Set the CMake prefix path
    set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/third_party" CACHE STRING "CMake prefix path")

    # Set the absl directory
    set(absl_DIR "${CMAKE_SOURCE_DIR}/third_party/install/lib/cmake/absl" CACHE STRING "Abseil CMake directory")

    # Set the utf8_range directory
    set(utf8_range_DIR "${CMAKE_SOURCE_DIR}/third_party/install/lib/cmake/utf8_range" CACHE STRING "utf8_range CMake directory")
    message(STATUS "Configuring for Android")
    # Android特定的设置
    add_definitions(-DPLATFORM_ANDROID)
    # Android可能需要特定的库路径和编译选项
#    set(OpenMP_C_FLAGS "-fopenmp")
#    set(OpenMP_CXX_FLAGS "-fopenmp")
#    set(OpenMP_omp_LIBRARY "omp")
    # Enable OpenMP
    find_package(OpenMP REQUIRED)
    if(OpenMP_CXX_FOUND)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif()
    # 设置 Armadillo 的路径
    set(ARMADILLO_LIBRARIES "${CMAKE_SOURCE_DIR}/third_party/arma/lib/libarmadillo.a")
    set(ARMADILLO_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/third_party/arma/include")
    set(Protobuf_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/third_party/install/include")
    set(Protobuf_LIBRARIES "${CMAKE_SOURCE_DIR}/third_party/install/lib/libprotobuf.a")
    # 找到Protobuf
    find_package(Protobuf REQUIRED)
    if (NOT Protobuf_FOUND)
        message(FATAL_ERROR "Could not find Protobuf")
    endif ()
    # 添加utf-8
    find_package(utf8_range HINTS ${CMAKE_SOURCE_DIR}/third_party/install/lib/cmake/utf8_range CONFIG REQUIRED)
    set(UTF8RANGE_LIBRARY
            utf8_range::utf8_validity
            utf8_range::utf8_range)
    # 添加abseil
    find_package(absl HINTS ${CMAKE_SOURCE_DIR}/third_party/install/lib/cmake/absl CONFIG REQUIRED)
    set(ABSEL_LIBARARY
            absl::absl_check
            absl::absl_log
            absl::algorithm
            absl::base
            absl::bind_front
            absl::bits
            absl::btree
            absl::cleanup
            absl::cord
            absl::core_headers
            absl::debugging
            absl::die_if_null
            absl::dynamic_annotations
            absl::flags
            absl::flat_hash_map
            absl::flat_hash_set
            absl::function_ref
            absl::hash
            absl::layout
            absl::log_initialize
            absl::log_severity
            absl::memory
            absl::node_hash_map
            absl::node_hash_set
            absl::optional
            absl::span
            absl::status
            absl::statusor
            absl::strings
            absl::synchronization
            absl::time
            absl::type_traits
            absl::utility
            absl::variant
    )
    # 添加onnx的文件路径
    set(ONNX_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/third_party/microsoft")
    set(ONNX_LIBRARIES "${CMAKE_SOURCE_DIR}/libs/onnx")
    # 加载glog路径
    set(GLOG_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/third_party/glog/include")
    set(GLOG_LIBRARIES "${CMAKE_SOURCE_DIR}/third_party/glog/lib/libglog.a")
else()
    # 查找OpenMP
    set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
    set(OpenMP_C_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.dylib")
    set(OpenMP_INCLUDE "/opt/homebrew/opt/libomp/include")
    message(STATUS "Configuring for non-Android platform")
    # 查找Armadillo库
    find_package(Armadillo REQUIRED)
    # 查找 protobuf 包
    find_package(Protobuf REQUIRED)
    # 查找abseil
    find_package(absl REQUIRED)
    # 查找 ONNX 包
    find_package(onnx REQUIRED)
    # 设置libs库
    set(ONNX_LIBRARIES "/usr/local/lib")
    # 设置abseil
    set(ABSEL_LIBARARY
            absl::base
            absl::strings
            absl::log
            absl::log_internal_check_op  # 可能需要这个
    )
    set(UTF8RANGE_LIBRARY "")

endif()

# 下载GoogleTest项目
include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)


# 查找OpenCV库
find_package(OpenCV REQUIRED)

# 导入google的Protobuf
include_directories(${Protobuf_INCLUDE_DIR})

# 启动测试
enable_testing()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/includes
        ${Protobuf_INCLUDE_DIRS}
        ${ONNX_INCLUDE_DIRS}
        ${GLOG_INCLUDE_DIRS}
        ${OpenMP_INCLUDE}
        ${ARMADILLO_INCLUDE_DIRS})

link_directories(${GLOG_LIBRARY_DIRS}
        ${ONNX_LIBRARIES})

add_executable(BatmanInfer main.cpp
        test/test_operator.cpp
        source/Tensor.cpp
        test/test_create_tensor.cpp
        test/tensor_get_values.cpp
        test/test_onnx.cpp
        source/others/utils.cpp
        source/onnx_conv/OnnxUtils.cpp
        source/ir.cpp
        source/runtime_attr.cpp
        source/runtime_op.cpp
        source/runtime_ir.cpp
        test/test_ir.cpp
        source/tensor_util.cpp
        test/test_topo.cpp
        source/operators/batman_operator.cpp
        test/test_ir_op.cpp
        source/layer/abstract/layer.cpp
        source/layer/abstract/layer_factory.cpp
        source/layer/abstract/param_layer.cpp
        test/test_layer_create.cpp
        test/test_sigmoid.cpp
        source/layer/detail/relu.cpp
        source/layer/detail/softmax.cpp
        source/layer/detail/sigmoid.cpp
        source/layer/detail/maxpooling.cpp
        test/test_maxpooling.cpp
        source/layer/detail/convolution.cpp
        test/test_convolution.cpp
        test/test_resnt.cpp
        source/layer/detail/add.cpp
        source/layer/detail/global_average_pool.cpp
        source/layer/detail/flatten.cpp
        source/layer/detail/gemm.cpp
        test/test_operators.cpp)

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

if(ANDROID)
    target_link_libraries(BatmanInfer OpenMP::OpenMP_CXX)
endif ()

# 添加测试
include(GoogleTest)
#gtest_discover_tests(BatmanInfer)

# 包含Armadillo头文件路径
target_include_directories(BatmanInfer PRIVATE ${ARMADILLO_INCLUDE_DIRS})
