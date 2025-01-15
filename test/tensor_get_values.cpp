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

    const BITensorShape input_shape(3, 2);
    const BITensorShape output_shape(2, 3);

    input.allocator()->init(BITensorInfo(input_shape, 1, BIDataType::U8));
    output.allocator()->init(BITensorInfo(output_shape, 1, BIDataType::U8));

    BINETranspose transpose;
    transpose.configure(&input, &output);

    input.allocator()->allocate();
    output.allocator()->allocate();

    // 填充输入张量数据
    uint8_t input_data[] = {1, 2, 3, 4, 5, 6}; // 3x2 矩阵
    std::memcpy(input.buffer(), input_data, sizeof(input_data));

    // 执行转置
    transpose.run();

    BIIOFormatInfo format;
    format.element_delim = ", ";  // 元素之间用逗号分隔
    format.row_delim = "\n";      // 每行换行
    format.align_columns = 1;     // 对齐列

    // 读取输出张量数据
    uint8_t output_data[6];
    std::memcpy(output_data, output.buffer(), sizeof(output_data));

    // 打印张量
    std::cout << "Output matrix:" << std::endl;
    for (size_t i = 0; i < output_shape[1]; ++i) {
        for (size_t j = 0; j < output_shape[0]; ++j) {
            std::cout << static_cast<int>(output_data[i * output_shape[0] + j]) << " ";
        }
        std::cout << std::endl;
    }
//    output.print(std::cout, format);

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

