//
// Created by Mason on 2024/10/11.
//

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <runtime/bi_tensor.hpp>
#include <runtime/bi_blob_lifetime_manager.hpp>
#include <runtime/bi_pool_manager.hpp>
#include <data/core/bi_tensor_info.hpp>
#include <cpu/kernels/assembly/bi_nd_range.hpp>
#include <runtime/bi_i_scheduler.hpp>
#include <mutex>
#include <data/core/bi_error.h>
#include <data/core/bi_vlidate.hpp>

#include <thread>

// 临时数据
#include <cpu/operators/internal/cpu_gemm_assembly_dispatch.hpp>
#include <runtime/neon/bi_ne_scheduler.hpp>
#include <runtime/neon/functions/bi_ne_transpose.hpp>
#include <runtime/neon/functions/bi_ne_reshape_layer.hpp>
#include <data/core/utils/misc/bi_shape_calculator.hpp>
#include <cpu/kernels/bi_cpu_gemm_inter_leave_4x4_kernel.hpp>
#include <runtime/neon/functions/bi_ne_gemm.hpp>
#include <runtime/neon/functions/bi_ne_split.hpp>
#include "runtime/neon/functions/bi_ne_mat_mul.hpp"
#include "runtime/neon/functions/bi_NESoftmaxLayer.h"
#include "data/core/utils/quantization/asymm_helpers.hpp"
#include <function_info/bi_MatMulInfo.h>
#include <runtime/neon/functions/ne_pixel_wise_multiplication.hpp>
#include <runtime/neon/functions/bi_ne_gemm_lowp_matrix_mul_core.hpp>
#include <runtime/neon/functions/bi_NEFullyConnectedLayer.h>


TEST(test_tensor_values, tensor_values1) {
    using namespace BatmanInfer;

    // 创建内存管理器组件
    // 我们需要两个内存管理器:
    // 1. 管理函数内的张量 (mm_layers)
    //    mm_layers：管理函数内部张量
    //* 职责：
    //      负责分配和释放函数内部的临时张量内存。
    //      这些张量的生命周期较短，通常在函数执行完成后即可释放。
    //* 优化目标：
    //      减少函数内部的内存分配和释放开销。
    //      通过复用内存，避免频繁的动态分配操作，提升性能。
    // 2. 管理函数的输入和输出张量 (mm_transitions)
    // * 职责：
    //      负责分配和管理函数之间传递的张量内存。
    //      这些张量需要在多个函数之间共享，因此其生命周期较长。
    // * 优化目标：
    //      确保输入和输出张量的内存可以在不同函数之间高效传递。
    //      避免不必要的内存拷贝和重新分配。

    // 创建生命周期管理器
    auto lifetime_mgr0 = std::make_shared<BIBlobLifetimeManager>();
    auto lifetime_mgr1 = std::make_shared<BIBlobLifetimeManager>();

    // 创建池化管理器
    auto pool_mgr0 = std::make_shared<BIPoolManager>();
    auto pool_mgr1 = std::make_shared<BIPoolManager>();

    // 创建内存管理器
    auto mm_layers = std::make_shared<BIMemoryManagerOnDemand>(lifetime_mgr0, pool_mgr0);
    auto mm_transitions = std::make_shared<BIMemoryManagerOnDemand>(lifetime_mgr1, pool_mgr1);

    // Initialize src tensor
    constexpr unsigned int width_src_image = 32;
    constexpr unsigned int height_src_image = 32;
    constexpr unsigned int ifm_src_img = 1;

    const BITensorShape src_shape(width_src_image, height_src_image, ifm_src_img);

    BITensor weights0{};
    BITensor weights1{};

    // 初始化conv0的张量信息
    constexpr unsigned int kernel_x_conv0 = 5;
    constexpr unsigned int kernel_y_conv0 = 5;
    constexpr unsigned int ofm_conv0 = 8;

    const BITensorShape weights_shape_conv0(kernel_x_conv0, kernel_y_conv0, src_shape.z(), ofm_conv0);

    weights0.allocator()->init(BITensorInfo(weights_shape_conv0, 1, BIDataType::F32));
//    weights0.print(std::cout);

    std::cout << "Hello" << std::endl;
}

TEST(test_tensor_values, nd_range_test) {
    // 行主序的填充方式
    using namespace BatmanGemm;
    // 创建一个 3D  范围，尺寸为 4x3x2
    BINDRange<3> range(4U, 3U, 2U);

    // 获取范围的迭代器，遍历[5, 10)
    auto it = range.iterator(5, 10);

    // 遍历范围
    while (!it.done()) {
        // 打印当前线性索引和对应的多维坐标
        std::cout << "Linear index: " << it.dim(0)
                  << ", dim(0): " << it.dim(0)
                  << ", dim(1): " << it.dim(1)
                  << ", dim(2): " << it.dim(2) << std::endl;

        // 打印 dim0_max的值
        std::cout << "dim0_max: " << it.dim0_max() << std::endl;

        it.next_dim0();
    }
}

std::mutex cout_mutex;

void matrix_workload(const BatmanInfer::ThreadInfo &info,
                     const std::vector<int> &matrix,
                     int rows, int cols) {
    // 每个线程处理的行数
    int chunk_size = rows / info.num_threads;
    int start_row = info.thread_id * chunk_size;
    int end_row = (info.thread_id == info.num_threads - 1) ? rows : start_row + chunk_size;

    for (int i = start_row; i < end_row; ++i)
        for (int j = 0; j < cols; ++j) {
            std::lock_guard<std::mutex> lock(cout_mutex);
            std::cout << "Thread " << info.thread_id << " processing element (" << i << ", " << j << ")\n";
        }
}

/**
 * @brief 测试Workload(工作负载信息)
 */
TEST(test_tensor_values, work_load_test) {
    int rows = 6, cols = 4, num_threads = 3;
    // 假设一个简单的矩阵
    std::vector<int> matrix(rows * cols, 1);

    std::vector<std::thread> threads;

    for (int i = 0; i < num_threads; ++i) {
        BatmanInfer::ThreadInfo info{i, num_threads};
        threads.emplace_back([=, &matrix]() { matrix_workload(info, matrix, rows, cols); });
    }

    // 等待所有线程完成
    for (auto &t: threads) {
        t.join();
    }

}

struct Params {
    unsigned int M;
    unsigned int N;
    unsigned int K;
    unsigned int batches;
    unsigned int multis;
    unsigned int sections;
    bool indirect;
};

/**
 * @brief 提取参数
 * @param a
 * @param b
 * @param d
 * @param info
 * @return
 */
Params extract_parameters(const BatmanInfer::BIITensorInfo *a,
                          const BatmanInfer::BIITensorInfo *b,
                          const BatmanInfer::BIITensorInfo *d,
                          const BatmanInfer::cpu::BIAsmGemmInfo &info) {
    BI_COMPUTE_ERROR_ON_NULLPTR(a, b, d);
    Params p{/* M */ static_cast<unsigned int>(d->tensor_shape().y()),
            /* N */ static_cast<unsigned int>(d->tensor_shape().x()),
            /* K */ static_cast<unsigned int>(a->tensor_shape().x()),
            /* batches */ 1,
            /* multis */ 1,
            /* sections */ 1,
            /* indirect */ false};

    if (info.method == BatmanInfer::cpu::BIAsmConvMethod::Conv ||
        info.method == BatmanInfer::cpu::BIAsmConvMethod::Indirect) {
        p.indirect = true;
        p.sections = b->tensor_shape()[2] * b->tensor_shape()[3];
    } else {
        p.multis = b->tensor_shape().z();
        p.batches = d->tensor_shape().total_size_upper(2) / p.multis;
    }

    // 更新M如果GEMM3D作为输出
    if (info.depth_output_gemm3d != 0) {
        p.M = d->tensor_shape().y() * d->tensor_shape().z();
        p.batches = d->tensor_shape().total_size_upper(3) / p.multis;
    }
    return p;
}

TEST(test_tensor_values, extract_params) {
    using namespace BatmanInfer;
    // 1. 创建输入张量A的信息
    BITensorShape shape_a(3, 4);
    BIDataType data_type_a = BIDataType::QASYMM8;
    BITensorInfo tensor_a(shape_a, 1, data_type_a);

    // 2. 创建输入张量 B 的信息
    BITensorShape shape_b(4, 2, 2, 2);  // B 的形状为 4 × 2 × 2 × 2
    BIDataType data_type_b = BIDataType::QASYMM8;  // 假设 B 的数据类型为 FLOAT32
    BITensorInfo tensor_b(shape_b, 1, data_type_b);

    // 3. 创建输出张量 D 的信息
    BITensorShape shape_d(3, 2, 2);  // D 的形状为 3 × 2 × 2
    BIDataType data_type_d = BIDataType::QASYMM8;  // 假设 D 的数据类型为 FLOAT32
    BITensorInfo tensor_d(shape_d, 1, data_type_d);

    // 4. 创建配置信息
    cpu::BIAsmGemmInfo info;
    info.method = cpu::BIAsmConvMethod::Conv;  // 设置为普通卷积方法
    info.depth_output_gemm3d = false;  // 不使用 3D GEMM 输出

    // 5. 调用 extract_parameters 函数
    Params params = extract_parameters(&tensor_a, &tensor_b, &tensor_d, info);

    // 6. 打印结果
    std::cout << "Extracted Parameters:" << std::endl;
    std::cout << "M: " << params.M << std::endl;
    std::cout << "N: " << params.N << std::endl;
    std::cout << "K: " << params.K << std::endl;
    std::cout << "Batches: " << params.batches << std::endl;
    std::cout << "Multis: " << params.multis << std::endl;
    std::cout << "Sections: " << params.sections << std::endl;
    std::cout << "Indirect: " << (params.indirect ? "true" : "false") << std::endl;
}

using namespace BatmanInfer;

BIIScheduler::Hints scheduling_hint_heuristic(BatmanGemm::GemmMethod method,
                                              BIDataType data_type) {
    // 调度汇编内核
    // 调度粒度的阈值，值为 200
    const int granule_threshold = 200;
    // 默认的调度提示(默认在X轴上并行)
    auto scheduling_hint = BIIScheduler::Hints(BIWindow::DimX);
    if (method == BatmanGemm::GemmMethod::GEMM_INTERLEAVED && data_type == BIDataType::F32)
        scheduling_hint = BIIScheduler::Hints(BIWindow::DimX,
                                              BIIScheduler::BIStrategyHint::DYNAMIC,
                                              granule_threshold);
    else if (method == BatmanGemm::GemmMethod::GEMM_INTERLEAVED_2D && (data_type == BIDataType::F32 ||
                                                                       data_type == BIDataType::F16 ||
                                                                       data_type == BIDataType::U8 ||
                                                                       data_type == BIDataType::S8))
        scheduling_hint = BIIScheduler::Hints(BIIScheduler::split_dimensions_all,
                                              BIIScheduler::BIStrategyHint::STATIC,
                                              granule_threshold);
    else if (method == BatmanGemm::GemmMethod::QUANTIZE_WRAPPER_2D && (data_type == BIDataType::QASYMM8 ||
                                                                       data_type ==
                                                                       BIDataType::QASYMM8_SIGNED))
        scheduling_hint = BIIScheduler::Hints(BIIScheduler::split_dimensions_all,
                                              BIIScheduler::BIStrategyHint::STATIC,
                                              granule_threshold);

    return scheduling_hint;
}

TEST(SchedulingHintHeuristicTest, GEMMInterleavedF32) {
    BatmanGemm::GemmMethod method = BatmanGemm::GemmMethod::GEMM_INTERLEAVED;
    BIDataType data_type = BIDataType::F32;

    // 调用函数
    auto result = scheduling_hint_heuristic(method, data_type);
    // 验证返回值
    EXPECT_EQ(result.split_dimension(), BIWindow::DimX); // 验证并行维度
    EXPECT_EQ(result.strategy(), BIIScheduler::BIStrategyHint::DYNAMIC); // 验证策略
    EXPECT_EQ(result.threshold(), 200); // 验证粒度阈值
}

TEST(SchedulingHintHeuristicTest, GEMMInterleaved2DF16) {
    BatmanGemm::GemmMethod method = BatmanGemm::GemmMethod::GEMM_INTERLEAVED_2D;
    BIDataType data_type = BIDataType::F16;

    // 调用函数
    auto result = scheduling_hint_heuristic(method, data_type);
    // 验证返回值
    EXPECT_EQ(result.split_dimension(), BIIScheduler::split_dimensions_all); // 验证并行维度
    EXPECT_EQ(result.strategy(), BIIScheduler::BIStrategyHint::STATIC); // 验证策略
    EXPECT_EQ(result.threshold(), 200); // 验证粒度阈值
}

TEST(BIIteratorTest, TestIteratorInit) {
    size_t num_dims = 3;
    // 每个维度的步幅
    BIStrides strides = {20, 5, 1};
    BIWindow window;

    // 设置维度 0 的起始位置为 1，结束位置为 10，步长为 2
    window.set(BIWindow::DimX, BIWindow::BIDimension(1, 10, 2));

    // 设置维度 1 的起始位置为 0，结束位置为 5，步长为 1
    window.set(BIWindow::DimY, BIWindow::BIDimension(0, 5, 1));

    // 设置维度 2 的起始位置为 2，结束位置为 8，步长为 1
    window.set(BIWindow::DimZ, BIWindow::BIDimension(2, 8, 1));

    // 创建一个模拟缓冲区
    // 假设缓冲区大小为 100 字节
    std::vector<uint8_t> buffer(100);
    // 偏移量为0
    size_t offset = 0;

    BIIterator iterator{num_dims, strides, buffer.data(), offset, window};

    // 打印结果以验证
    for (size_t i = 0; i < num_dims; ++i) {
        std::cout << "Dim " << i << ":\n";
//        std::cout << "  Stride: " << iterator._dims[i]._stride << "\n";
//        std::cout << "  Start: " << iterator._dims[i]._dim_start << "\n";
    }
}

TEST(BITensorTest, transpose_test) {
    // 定义输入和输出张量
    BITensor input, output;

    const BITensorShape input_shape(12, 64);
    const BITensorShape output_shape(64, 12);

    input.allocator()->init(BITensorInfo(input_shape, 1, BIDataType::U8));
    output.allocator()->init(BITensorInfo(output_shape, 1, BIDataType::U8));

    BINETranspose transpose;
    transpose.configure(&input, &output);

    input.allocator()->allocate();
    output.allocator()->allocate();

    // 填充输入张量数据
//    uint8_t input_data[] = {1, 2, 3, 4, 5, 6}; // 3x2 矩阵
//    std::memcpy(input.buffer(), input_data, sizeof(input_data));

    // 执行转置
    transpose.run();

    BIIOFormatInfo format;
    format.element_delim = ", ";  // 元素之间用逗号分隔
    format.row_delim = "\n";      // 每行换行
    format.align_columns = 1;     // 对齐列

    output.print(std::cout, format);

}


TEST(BITensorTest, reshape_test) {
    BITensor input_tensor, output_tensor;

    BITensorShape input_shape(4, 4, 1);

    BITensorShape output_shape(2, 8, 1);

    input_tensor.allocator()->init(BITensorInfo(input_shape, 1, BIDataType::U8));
    output_tensor.allocator()->init(BITensorInfo(output_shape, 1, BIDataType::U8));

    BINEReshapeLayer reshape_layer;

    reshape_layer.configure(&input_tensor, &output_tensor);

    input_tensor.allocator()->allocate();
    output_tensor.allocator()->allocate();

    // 填充输入张量数据（示例：填充为 1 到 16）
    uint8_t *input_data = reinterpret_cast<uint8_t *>(input_tensor.buffer());
    for (int i = 0; i < 16; ++i) {
        input_data[i] = static_cast<uint8_t>(i + 1);
    }

    reshape_layer.run();

    uint8_t *output_data = reinterpret_cast<uint8_t *>(output_tensor.buffer());
    BIIOFormatInfo format;
    format.element_delim = ", ";  // 元素之间用逗号分隔
    format.row_delim = "\n";      // 每行换行
    format.align_columns = 1;     // 对齐列

//    output_tensor.print(std::cout, format);
    for (int i = 0; i < 16; ++i) {
        std::cout << static_cast<int>(output_data[i]) << " ";
    }
//    output_tensor.print(std::cout);
    std::cout << std::endl;
}

TEST(BITensorTest, DefaultInterleave) {
    BITensor input_tensor;
    BITensorShape input_shape{8, 8};

    input_tensor.allocator()->init(BITensorInfo(input_shape, 1, BIDataType::U8));
    input_tensor.allocator()->allocate();

    auto result = BatmanInfer::misc::shape_calculator::compute_interleaved_shape(*input_tensor.info());
    EXPECT_EQ(result.x(), 32);
    EXPECT_EQ(result.y(), 2);
}

TEST(BITensorTest, CustomInterleaveHeight) {
    BITensor input_tensor;
    BITensorShape input_shape{10, 8};

    input_tensor.allocator()->init(BITensorInfo(input_shape, 1, BIDataType::U8));
    input_tensor.allocator()->allocate();

    auto result = BatmanInfer::misc::shape_calculator::compute_interleaved_shape(*input_tensor.info(), 2);
    EXPECT_EQ(result.x(), 80);
    EXPECT_EQ(result.y(), 1);
}

TEST(BITensorTest, GEMMIntervalTest) {
    using namespace BatmanInfer;
    // 定义输入和输出张量的形状
    BITensorShape src_shape(16, 16);  // 假设输入是16x16矩阵

    // 创建输入和输出张量
    BITensor src, dst;

    // 配置输入张量
    src.allocator()->init(BITensorInfo(src_shape, 1, BIDataType::F32));  // 假设数据类型为F32

    // 分配张量内存
    src.allocator()->allocate();
//    dst.allocator()->allocate();
    ::cpu::kernels::BICpuGemmInterleave4x4Kernel kernel;
    kernel.configure(src.info(), dst.info());


}

void interleave_B_inplace(const float *B, float *B_interleave) {
    // Load four rows of B
    float32x4_t b_row0 = vld1q_f32(B);      // Load [b11, b12, b13, b14]
    float32x4_t b_row1 = vld1q_f32(B + 4);  // Load [b21, b22, b23, b24]
    float32x4_t b_row2 = vld1q_f32(B + 8);  // Load [b31, b32, b33, b34]
    float32x4_t b_row3 = vld1q_f32(B + 12); // Load [b41, b42, b43, b44]

    // Transpose using NEON intrinsics
    float32x4x4_t b_transposed = vld4q_f32(B);

    // Store transposed result back to original memory
    vst1q_f32(B_interleave, b_transposed.val[0]); // Store [b11, b21, b31, b41]
    vst1q_f32(B_interleave + 4, b_transposed.val[1]); // Store [b12, b22, b32, b42]
    vst1q_f32(B_interleave + 8, b_transposed.val[2]); // Store [b13, b23, b33, b43]
    vst1q_f32(B_interleave + 12, b_transposed.val[3]); // Store [b14, b24, b34, b44]
}

// Helper function to print matrix
void print_matrix(const char *name, const float *mat, int rows, int cols) {
    std::cout << name << ":\n";
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << std::setw(8) << mat[i * cols + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\nMemory layout: ";
    for (int i = 0; i < rows * cols; ++i) {
        std::cout << mat[i] << " ";
    }
    std::cout << "\n\n";
}

TEST(BITensor, GEMMInterval) {
    const int rows = 4, cols = 4;
    float B[rows * cols] = {
            1, 2, 3, 4,    // B = [1  2  3  4]
            5, 6, 7, 8,    //     [5  6  7  8]
            9, 10, 11, 12,   //     [9  10 11 12]
            13, 14, 15, 16    //     [13 14 15 16]
    };
    float B_interleaved[rows * cols] = {0};

    print_matrix("Matrix B (before interleave)", B, rows, cols);

    interleave_B_inplace(B, B_interleaved);

    print_matrix("Matrix B (after interleave)", B_interleaved, rows, cols);
}

TEST(BITensor, NEGEMM_exmaple_01) {
    // Basic using
    using namespace BatmanInfer;

    BatmanInfer::BIScheduler::get().set_num_threads(4);

    // 输入张量 A, B, C 和输出张量D
    BITensor a, b, c, d;

    // 配置张量的形状 (假设矩阵 A 是 MxK，矩阵 B 是 KxN）
    const unsigned int M = 4; // 矩阵A的行数
    const unsigned int K = 3; // 矩阵A的列数，矩阵B的行数
    const unsigned int N = 2; // 矩阵B的列数

    // 配置张量的形状和数据类型
    BITensorInfo a_info(BITensorShape(K, M), 1, BIDataType::F32);  // 矩阵 A
    BITensorInfo b_info(BITensorShape(N, K), 1, BIDataType::F32);  // 矩阵 B
    BITensorInfo c_info(BITensorShape(N, M), 1, BIDataType::F32);  // 矩阵 C（可选）
    BITensorInfo d_info(BITensorShape(N, M), 1, BIDataType::F32);  // 输出矩阵 D

    // 初始化张量
    a.allocator()->init(a_info);
    b.allocator()->init(b_info);
    c.allocator()->init(c_info);
    d.allocator()->init(d_info);

    // 3. 配置BINEGEMM函数
    BINEGEMM gemm;

    // 配置 NEGEMM 函数
    float alpha = 1.0f;  // 矩阵乘积的权重
    float beta = 0.1f;   // 矩阵 C 的权重（如果不需要 C，可以设置为 0）

    GEMMInfo gemm_info(false, false, true /* enable reshaping */, 0, 0, true /* reshape only on first run */);

    gemm.configure(&a, &b, &c, &d, alpha, beta, gemm_info);

    // 分配内存(为张量分配内存)
    a.allocator()->allocate();
    b.allocator()->allocate();
    c.allocator()->allocate();
    d.allocator()->allocate();

    // 示例：填充张量 A 的数据
    auto a_data = reinterpret_cast<float *>(a.buffer());
    for (unsigned int i = 0; i < M * K; ++i) {
        a_data[i] = static_cast<float>(i);  // 填充一些测试数据
    }
    auto b_data = reinterpret_cast<float *>(b.buffer());
    for (unsigned int i = 0; i < K * N; ++i) {
        b_data[i] = static_cast<float>(1);  // 填充一些测试数据
    }
    auto c_data = reinterpret_cast<float *>(c.buffer());
    for (unsigned int i = 0; i < M * N; ++i) {
        c_data[i] = static_cast<float>(1);  // 填充一些测试数据
    }

    BIIOFormatInfo format;
    format.element_delim = ", ";  // 元素之间用逗号分隔
    format.row_delim = "\n";      // 每行换行
    format.align_columns = 1;     // 对齐列

    // 打印数据
//    a.print(std::cout, format);
//    b.print(std::cout, format);

    gemm.run();
    const BITensorShape output_shape(N, M);

    // 访问输出数据
//    d.print(std::cout, format);
    float output_data[M * N];
    std::memcpy(output_data, d.buffer(), sizeof(output_data));

    // 打印张量
    std::cout << "Output matrix:" << std::endl;
    for (size_t i = 0; i < output_shape[1]; ++i) {
        for (size_t j = 0; j < output_shape[0]; ++j) {
            std::cout << static_cast<float >(output_data[i * output_shape[0] + j]) << " ";
        }
        std::cout << std::endl;
    }
}

TEST(BITensor, NESplit_example_02) {
    // 创建输入张量
    BITensor input_tensor;
    const BITensorShape input_shape(2, 4);
    input_tensor.allocator()->init(BITensorInfo(input_shape, 1, BIDataType::F32));

    // Step 2: 创建输出张量列表（注意：需要声明为 const）
    const unsigned int num_splits = 4; // 分割成 4 个张量
    std::vector<BITensor> output_tensors(num_splits);
    for (auto &tensor: output_tensors) {
        BITensorShape output_shape = input_shape;
        output_shape.set(1, input_shape[1] / num_splits); // 修改分割维度的大小
        tensor.allocator()->init(BITensorInfo(output_shape, 1, BIDataType::F32));
    }

    // 将 output_tensors 转换为 Tensor* 数组
    std::vector<BIITensor *> output_tensor_ptrs(num_splits);
    for (unsigned int i = 0; i < num_splits; ++i) {
        output_tensor_ptrs[i] = &output_tensors[i];
    }

    // Step 3: 配置 NESplit
    BINESplit split_layer;
    split_layer.configure(&input_tensor, output_tensor_ptrs, 1); // 按第二个维度分割

    // Step 4: 分配内存
    input_tensor.allocator()->allocate();
    for (auto &tensor: output_tensors) {
        tensor.allocator()->allocate();
    }

    BIIOFormatInfo format;
    format.element_delim = ", ";  // 元素之间用逗号分隔
    format.row_delim = "\n";      // 每行换行
    format.align_columns = 1;     // 对齐列

    // Step 5: 运行 NESplit
    split_layer.run();

    std::cout << "NESplit executed successfully!" << std::endl;

    for (auto &tensor: output_tensors) {
//        tensor.print(std::cout, format);
    }
}

void print_tensor_qasymm8(const BITensor &tensor) {
    // 获取张量的量化信息
    auto quant_info = tensor.info()->quantization_info();
    float scale = quant_info.uniform().scale;  // 量化比例因子
    int32_t offset = quant_info.uniform().offset; // 量化零点

    // 获取张量形状
    const BITensorShape &shape = tensor.info()->tensor_shape();
    size_t width = shape[0];
    size_t height = shape[1];

    // 获取张量数据指针
    const uint8_t *data = reinterpret_cast<const uint8_t *>(tensor.buffer());

    // 打印张量数据（反量化为浮点值）
    std::cout << "BITensor (QASYMM8, dequantized):" << std::endl;
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            // 反量化公式：real_value = scale * (quantized_value - offset)
            float real_value = scale * (static_cast<int32_t>(data[y * width + x]) - offset);
            std::cout << std::fixed << std::setprecision(4) << real_value << " ";
        }
        std::cout << std::endl;
    }
}

TEST(BITensor, NEMatMul_example_01) {
    // 定义输入和输出张量的形状
    BITensorShape shape_a(3, 2); // 左矩阵 (3x2)
    BITensorShape shape_b(4, 2); // 右矩阵 (4x2)，需要转置为 (2x4)
    BITensorShape shape_c(4, 3); // 输出矩阵 (4x3)

    // 创建输入和输出张量
    BITensor tensor_a, tensor_b, tensor_c;

    // 配置张量
    tensor_a.allocator()->init(BITensorInfo(shape_a, 1, BIDataType::QASYMM8, BIQuantizationInfo(0.5f, 128)));
    tensor_b.allocator()->init(BITensorInfo(shape_b, 1, BIDataType::QASYMM8, BIQuantizationInfo(0.25f, 128)));
    tensor_c.allocator()->init(BITensorInfo(shape_c, 1, BIDataType::QASYMM8, BIQuantizationInfo(0.125f, 128)));

    tensor_a.info()->set_are_values_constant(false);
    tensor_b.info()->set_are_values_constant(false);
    // 定义 MatMul 配置信息
    BIMatMulInfo matmul_info; // 不转置左矩阵，转置右矩阵
    matmul_info.adj_lhs(true).adj_rhs(false);
    BICpuMatMulSettings settings;
    settings.fast_math(true); // 启用快速数学模式

    // 定义激活函数信息（可选）
//    BIActivationLayerInfo act_info(BIActivationLayerInfo::ActivationFunction::RELU);

    // 创建 MatMul 操作对象
    BINEMatMul matmul;

    // 配置 MatMul 操作
    matmul.configure(&tensor_a, &tensor_b, &tensor_c, matmul_info, settings);

    // 分配内存
    tensor_a.allocator()->allocate();
    tensor_b.allocator()->allocate();
    tensor_c.allocator()->allocate();

    // 填充输入张量数据
    auto a_ptr = tensor_a.buffer();
    auto b_ptr = tensor_b.buffer();
    for (int i = 0; i < shape_a.total_size(); ++i) {
        a_ptr[i] = 128 + i; // 示例数据
    }
    for (int i = 0; i < shape_b.total_size(); ++i) {
        b_ptr[i] = 128 + 2 * i; // 示例数据
    }

    print_tensor_qasymm8(tensor_a);
}

TEST(BITensor, NEMul_example_01) {
    using namespace BatmanInfer;
    // 1. 创建张量形状和量化参数
    BITensorShape shape(32, 32);  // 示例大小为32x32

    // 初始化张量信息
    BITensorInfo tensor_info(shape, 1, BIDataType::F32);

    // 创建输入和输出张量
    BITensor src1, src2, dst;
    src1.allocator()->init(tensor_info);
    src2.allocator()->init(tensor_info);
    dst.allocator()->init(tensor_info);

    // 分配内存
    src1.allocator()->allocate();
    src2.allocator()->allocate();
    dst.allocator()->allocate();

    // 填充输入张量的数据
    float *src1_ptr = reinterpret_cast<float *>(src1.buffer());
    float *src2_ptr = reinterpret_cast<float *>(src2.buffer());
    for (size_t i = 0; i < shape.total_size(); ++i) {
        src1_ptr[i] = static_cast<float>(i + 1); // src1: 1, 2, 3, ...
        src2_ptr[i] = static_cast<float>(2);     // src2: 全部填充为 2
    }

    BINEPixelWiseMultiplication multiply;
    multiply.configure(&src1, &src2, &dst, 1.0f, BIConvertPolicy::SATURATE, BIRoundingPolicy::TO_ZERO);

    // 运行算子
    multiply.run();

    // 验证输出张量的数据
    float *dst_ptr = reinterpret_cast<float *>(dst.buffer());
    for (size_t i = 0; i < shape.total_size(); ++i)
        EXPECT_FLOAT_EQ(dst_ptr[i], src1_ptr[i] * src2_ptr[i]); // 验证结果

    BIIOFormatInfo format;
    format.element_delim = ", ";  // 元素之间用逗号分隔
    format.row_delim = "\n";      // 每行换行
    format.align_columns = 1;     // 对齐列

    dst.print(std::cout, format);
}

TEST(NESoftmaxLayerTest, BasicSoftmaxTest) {
    using namespace BatmanInfer;
    // 输入和输出张量
    BITensor input, output;

    // 定义张量形状 (假设批量大小为 1，特征数为 4)
    const BITensorShape shape(4); // 形状为 [4]

    // 配置输入张量
    input.allocator()->init(BITensorInfo(shape, 1, BIDataType::F32)); // 单精度浮点数
    output.allocator()->init(BITensorInfo(shape, 1, BIDataType::F32));

    // 创建 Softmax 层
    BINESoftmaxLayer softmax_layer;
    softmax_layer.configure(&input, &output);

    // 分配内存
    input.allocator()->allocate();
    output.allocator()->allocate();

    // 填充输入数据
    float input_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    std::memcpy(input.buffer(), input_data, sizeof(input_data));

    // 运行 Softmax 层
    softmax_layer.run();

    // 检查输出
    float *output_data = reinterpret_cast<float *>(output.buffer());
    float expected_output[] = {
            std::exp(1.0f) / (std::exp(1.0f) + std::exp(2.0f) + std::exp(3.0f) + std::exp(4.0f)),
            std::exp(2.0f) / (std::exp(1.0f) + std::exp(2.0f) + std::exp(3.0f) + std::exp(4.0f)),
            std::exp(3.0f) / (std::exp(1.0f) + std::exp(2.0f) + std::exp(3.0f) + std::exp(4.0f)),
            std::exp(4.0f) / (std::exp(1.0f) + std::exp(2.0f) + std::exp(3.0f) + std::exp(4.0f)),
    };

    // 验证输出是否与预期一致
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(output_data[i], expected_output[i], 1e-5);
    }
    BIIOFormatInfo format;
    format.element_delim = ", ";  // 元素之间用逗号分隔
    format.row_delim = "\n";      // 每行换行
    format.align_columns = 1;     // 对齐列

    std::cout << "Output matrix:" << std::endl;
    output.print(std::cout, format);
}

// 辅助函数：填充 Tensor 数据
// 简单的填充函数
// 使用 buffer() 填充 Tensor 数据
void fill_tensor_buffer(BITensor &tensor, BIDataType data_type, uint8_t fill_value = 1) {
    // 获取底层内存指针
    auto buffer_ptr = tensor.buffer();

    // 获取 Tensor 的总大小
    const size_t total_size = tensor.info()->total_size();

    // 填充数据
    if (data_type == BIDataType::QASYMM8) {
        // 填充 QASYMM8 类型的张量
        for (size_t i = 0; i < total_size; ++i) {
            reinterpret_cast<uint8_t *>(buffer_ptr)[i] = fill_value;
        }
    } else if (data_type == BIDataType::S32) {
        // 填充 S32 类型的张量
        for (size_t i = 0; i < total_size / sizeof(int32_t); ++i) {
            reinterpret_cast<int32_t *>(buffer_ptr)[i] = static_cast<int32_t>(fill_value);
        }
    } else if (data_type == BIDataType::F32) {
        // 填充 S32 类型的张量
        for (size_t i = 0; i < total_size / sizeof(float); ++i) {
            reinterpret_cast<float *>(buffer_ptr)[i] = static_cast<float >(fill_value);
        }
    } else {
        throw std::runtime_error("Unsupported data type for tensor filling!");
    }
}

TEST(BICpuLowpGemm, BasicGemmTest) {
    using namespace BatmanInfer;

    const BITensorShape input_shape(1024, 1024);    // 2x2
    const BITensorShape weights_shape(1024, 1024);  // 2x2
    const BITensorShape output_shape(1024, 1024);   // 2x2
    const BITensorShape bias_shape(1024);        // bias向量

    // 2. 创建张量
    BITensor input, weights, bias, output;

    // 3. 量化参数
    const BIQuantizationInfo quant_info(1.0f / 255.0f, 128);

    // 4. 配置张量信息
    BITensorInfo input_info(input_shape, 1, BIDataType::QASYMM8, quant_info);
    BITensorInfo weights_info(weights_shape, 1, BIDataType::QASYMM8, quant_info);
    BITensorInfo bias_info(bias_shape, 1, BIDataType::S32);
    BITensorInfo output_info(output_shape, 1, BIDataType::QASYMM8, quant_info);

    // 5. 初始化并分配内存
    input.allocator()->init(input_info);
    weights.allocator()->init(weights_info);
    bias.allocator()->init(bias_info);
    output.allocator()->init(output_info);

    input.allocator()->allocate();
    weights.allocator()->allocate();
    bias.allocator()->allocate();
    output.allocator()->allocate();

    // 6. 填充示例数据
    uint8_t *input_ptr = input.buffer();
    uint8_t *weights_ptr = weights.buffer();
    int32_t *bias_ptr = reinterpret_cast<int32_t *>(bias.buffer());

    uint8_t input_data[] = {10, 10, 10, 10};
    uint8_t weights_data[] = {1, 1, 1, 1};
    int32_t bias_data[] = {100, 100};

    memcpy(input_ptr, input_data, sizeof(input_data));
    memcpy(weights_ptr, weights_data, sizeof(weights_data));
    memcpy(bias_ptr, bias_data, sizeof(bias_data));

    // 7. 设置GEMMInfo，包括输出阶段
    GEMMInfo gemm_info;
    // 计算量化参数
    const int32_t input_offset = -quant_info.offset()[0];
    const int32_t weights_offset = -quant_info.offset()[0];
    const int32_t output_offset = quant_info.offset()[0];
    const float scale = quant_info.scale()[0];

    // 设置输出阶段
    BIGEMMLowpOutputStageInfo output_stage;
    output_stage.type = BIGEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT;
    output_stage.output_data_type = BIDataType::QASYMM8;
    output_stage.gemmlowp_offset = output_offset;
    // 计算multiplier和shift
    int32_t output_multiplier;
    int32_t output_shift;
    quantization::calculate_quantized_multiplier_less_than_one(
            scale, &output_multiplier, &output_shift);
    output_stage.gemmlowp_multiplier = output_multiplier;
    output_stage.gemmlowp_shift = output_shift;
    // 设置到gemm_info
    gemm_info.set_gemmlowp_output_stage(output_stage);

    // 8. 配置并运行GEMM
    BINEGEMMLowpMatrixMultipleCore gemm;
    gemm.configure(&input, &weights, nullptr, &output, gemm_info);

    // 记录开始时间点
    auto start = std::chrono::high_resolution_clock::now();

    // 调用需要测试的函数
    gemm.run();

    // 记录结束时间点
    auto end = std::chrono::high_resolution_clock::now();

    // 计算时间差
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // 打印结果
    std::cout << "Function execution time: " << duration.count() << " microseconds" << std::endl;
//    // 9. 打印结果
//    BIWindow window;
//    window.use_tensor_dimensions(output.info()->tensor_shape());
//
//    execute_window_loop(window, [&](const BICoordinates &id) {
//        uint8_t *output_ptr = output.buffer();
//        std::cout << static_cast<int>(output_ptr[id.y() * output_shape[0] + id.x()]) << " ";
//        if (id.x() == output_shape[0] - 1) std::cout << std::endl;
//    });
}

TEST(BICpuGemm, BasicGemmTest01) {
    using namespace BatmanInfer;
    // 1. 定义小矩阵: 2x2 * 2x2 = 2x2
    const BITensorShape input_shape(16, 768);    // 2x2
    const BITensorShape weights_shape(768, 2304);  // 2x2
    const BITensorShape output_shape(16, 2304);   // 2x2
    const BITensorShape bias_shape(2304);        // bias向量

// 2. 创建张量
    BITensor input, weights, bias, output;

// 3. 配置张量信息（使用F32而不是QASYMM8）
    BITensorInfo input_info(input_shape, 1, BIDataType::F32);
    BITensorInfo weights_info(weights_shape, 1, BIDataType::F32);
    BITensorInfo bias_info(bias_shape, 1, BIDataType::F32);
    BITensorInfo output_info(output_shape, 1, BIDataType::F32);

// 4. 初始化并分配内存
    input.allocator()->init(input_info);
    weights.allocator()->init(weights_info);
    bias.allocator()->init(bias_info);
    output.allocator()->init(output_info);

    input.allocator()->allocate();
    weights.allocator()->allocate();
    bias.allocator()->allocate();
    output.allocator()->allocate();

    // 5. 填充示例数据（使用之前反量化得到的实际值）
    float *input_ptr = reinterpret_cast<float *>(input.buffer());
    float *weights_ptr = reinterpret_cast<float *>(weights.buffer());
    float *bias_ptr = reinterpret_cast<float *>(bias.buffer());

    float input_data[] = {-0.463f, -0.463f, -0.463f, -0.463f};
    float weights_data[] = {-0.498f, -0.498f, -0.498f, -0.498f};
    float bias_data[] = {100.0f, 100.0f};

    memcpy(input_ptr, input_data, sizeof(input_data));
    memcpy(weights_ptr, weights_data, sizeof(weights_data));
    memcpy(bias_ptr, bias_data, sizeof(bias_data));

// 6. 配置并运行GEMM
    BINEGEMM gemm;
    gemm.configure(&weights, &input, &bias, &output, 1.0f, 1.0f);
    gemm.prepare();

    // 记录开始时间点
    auto start = std::chrono::high_resolution_clock::now();

    // 调用需要测试的函数
    gemm.run();

    // 记录结束时间点
    auto end = std::chrono::high_resolution_clock::now();

    // 计算时间差
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // 打印结果
    std::cout << "Function execution time: " << duration.count() << " microseconds" << std::endl;


//// 7. 打印结果
//    BIWindow window;
//    window.use_tensor_dimensions(output.info()->tensor_shape());
//
//    execute_window_loop(window, [&](const BICoordinates &id) {
//        float *output_ptr = reinterpret_cast<float *>(output.buffer());
//        std::cout << output_ptr[id.y() * output_shape[0] + id.x()] << " ";
//        if (id.x() == output_shape[0] - 1) std::cout << std::endl;
//    });
}

TEST(BIFullyConnected, BasicFullyTest) {
    // 输入张量维度 (batch_size, input_size)
    const BITensorShape input_shape(4096, 512); // 假设 batch_size=1, input_size=128

    // 输出张量维度 (batch_size, output_size)
    const BITensorShape output_shape(64, 512); // 假设 output_size=64

    // 权重张量维度 (output_size, input_size)
    const BITensorShape weights_shape(64, 4096);

    // 偏置张量维度 (output_size)
    const BITensorShape biases_shape(512);

    // 创建输入、权重、偏置和输出张量
    BITensor input, weights, biases, output;

    // 配置量化信息 (scale 和 zero_point)
    BIQuantizationInfo input_quant_info(0.1f, 128);   // 假设输入张量的 scale=0.1, zero_point=128
    BIQuantizationInfo weights_quant_info(0.05f, 128); // 假设权重张量的 scale=0.05, zero_point=128
    BIQuantizationInfo output_quant_info(0.2f, 128);  // 假设输出张量的 scale=0.2, zero_point=128

    // 初始化张量的数据类型和量化信息
    input.allocator()->init(BITensorInfo(input_shape, 1, BIDataType::QASYMM8, input_quant_info));
    weights.allocator()->init(BITensorInfo(weights_shape, 1, BIDataType::QASYMM8, weights_quant_info));
    biases.allocator()->init(BITensorInfo(biases_shape, 1, BIDataType::S32)); // 偏置通常是 S32 类型
    output.allocator()->init(BITensorInfo(output_shape, 1, BIDataType::QASYMM8, output_quant_info));

    // 创建 NEFullyConnectedLayer 实例
    BINEFullyConnectedLayer fc_layer;

    // 配置 FullyConnectedLayerInfo
    BIFullyConnectedLayerInfo fc_info;
    fc_info.transpose_weights = false; // 如果权重需要转置

    // 配置全连接层
    fc_layer.configure(&input, &weights, &biases, &output, fc_info);

    // 分配张量内存
    input.allocator()->allocate();
    weights.allocator()->allocate();
    biases.allocator()->allocate();
    output.allocator()->allocate();

    // 模拟数据填充 (实际中应加载量化后的数据)
    // 注意：这里的填充需要符合量化格式
    auto input_ptr = reinterpret_cast<uint8_t *>(input.buffer());
    for (size_t i = 0; i < input.info()->total_size(); ++i) {
        input_ptr[i] = 128; // 假设输入数据全为 zero_point
    }

    auto weights_ptr = reinterpret_cast<uint8_t *>(weights.buffer());
    for (size_t i = 0; i < weights.info()->total_size(); ++i) {
        weights_ptr[i] = 128; // 假设权重数据全为 zero_point
    }

    auto biases_ptr = reinterpret_cast<int32_t *>(biases.buffer());
    for (size_t i = 0; i < biases.info()->total_size() / sizeof(int32_t); ++i) {
        biases_ptr[i] = 0; // 偏置为零
    }

    auto start = std::chrono::high_resolution_clock::now();

    // 运行全连接层
    fc_layer.run();

    // 记录结束时间点
    auto end = std::chrono::high_resolution_clock::now();

    // 计算时间差
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // 打印结果
    std::cout << "Function execution time: " << duration.count() << " microseconds" << std::endl;

    BIIOFormatInfo format;
    format.element_delim = ", ";  // 元素之间用逗号分隔
    format.row_delim = "\n";      // 每行换行
    format.align_columns = 1;     // 对齐列

    output.print(std::cout, format);
}