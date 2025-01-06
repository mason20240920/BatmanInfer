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
    auto mm_layers      = std::make_shared<BIMemoryManagerOnDemand>(lifetime_mgr0, pool_mgr0);
    auto mm_transitions = std::make_shared<BIMemoryManagerOnDemand>(lifetime_mgr1, pool_mgr1);

    // Initialize src tensor
    constexpr unsigned int width_src_image  = 32;
    constexpr unsigned int height_src_image = 32;
    constexpr unsigned int ifm_src_img      = 1;

    const BITensorShape src_shape(width_src_image, height_src_image, ifm_src_img);

    BITensor weights0{};
    BITensor weights1{};

    // 初始化conv0的张量信息
    constexpr unsigned int kernel_x_conv0 = 5;
    constexpr unsigned int kernel_y_conv0 = 5;
    constexpr unsigned int ofm_conv0      = 8;

    const BITensorShape weights_shape_conv0(kernel_x_conv0, kernel_y_conv0, src_shape.z(), ofm_conv0);

    weights0.allocator()->init(BITensorInfo(weights_shape_conv0, 1, BIDataType::F32));
    weights0.print(std::cout);

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
    int start_row  = info.thread_id * chunk_size;
    int end_row    = (info.thread_id == info.num_threads - 1) ? rows : start_row + chunk_size;

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
    int              rows = 6, cols = 4, num_threads = 3;
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
    bool         indirect;
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
        p.multis  = b->tensor_shape().z();
        p.batches = d->tensor_shape().total_size_upper(2) / p.multis;
    }

    // 更新M如果GEMM3D作为输出
    if (info.depth_output_gemm3d != 0) {
        p.M       = d->tensor_shape().y() * d->tensor_shape().z();
        p.batches = d->tensor_shape().total_size_upper(3) / p.multis;
    }
    return p;
}

TEST(test_tensor_values, extract_params) {
    using namespace BatmanInfer;
    // 1. 创建输入张量A的信息
    BITensorShape shape_a(3, 4);
    BIDataType    data_type_a = BIDataType::QASYMM8;
    BITensorInfo  tensor_a(shape_a, 1, data_type_a);

    // 2. 创建输入张量 B 的信息
    BITensorShape shape_b(4, 2, 2, 2);  // B 的形状为 4 × 2 × 2 × 2
    BIDataType    data_type_b = BIDataType::QASYMM8;  // 假设 B 的数据类型为 FLOAT32
    BITensorInfo  tensor_b(shape_b, 1, data_type_b);

    // 3. 创建输出张量 D 的信息
    BITensorShape shape_d(3, 2, 2);  // D 的形状为 3 × 2 × 2
    BIDataType    data_type_d = BIDataType::QASYMM8;  // 假设 D 的数据类型为 FLOAT32
    BITensorInfo  tensor_d(shape_d, 1, data_type_d);

    // 4. 创建配置信息
    cpu::BIAsmGemmInfo info;
    info.method              = cpu::BIAsmConvMethod::Conv;  // 设置为普通卷积方法
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
    auto      scheduling_hint   = BIIScheduler::Hints(BIWindow::DimX);
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
    BatmanGemm::GemmMethod method    = BatmanGemm::GemmMethod::GEMM_INTERLEAVED;
    BIDataType             data_type = BIDataType::F32;

    // 调用函数
    auto result = scheduling_hint_heuristic(method, data_type);
    // 验证返回值
    EXPECT_EQ(result.split_dimension(), BIWindow::DimX); // 验证并行维度
    EXPECT_EQ(result.strategy(), BIIScheduler::BIStrategyHint::DYNAMIC); // 验证策略
    EXPECT_EQ(result.threshold(), 200); // 验证粒度阈值
}

TEST(SchedulingHintHeuristicTest, GEMMInterleaved2DF16) {
    BatmanGemm::GemmMethod method    = BatmanGemm::GemmMethod::GEMM_INTERLEAVED_2D;
    BIDataType             data_type = BIDataType::F16;

    // 调用函数
    auto result = scheduling_hint_heuristic(method, data_type);
    // 验证返回值
    EXPECT_EQ(result.split_dimension(), BIIScheduler::split_dimensions_all); // 验证并行维度
    EXPECT_EQ(result.strategy(), BIIScheduler::BIStrategyHint::STATIC); // 验证策略
    EXPECT_EQ(result.threshold(), 200); // 验证粒度阈值
}

TEST(BIIteratorTest, TestIteratorInit) {
    size_t    num_dims = 3;
    // 每个维度的步幅
    BIStrides strides  = {20, 5, 1};
    BIWindow  window;

    // 设置维度 0 的起始位置为 1，结束位置为 10，步长为 2
    window.set(BIWindow::DimX, BIWindow::BIDimension(1, 10, 2));

    // 设置维度 1 的起始位置为 0，结束位置为 5，步长为 1
    window.set(1, BIWindow::BIDimension(0, 5, 1));

    // 设置维度 2 的起始位置为 2，结束位置为 8，步长为 1
    window.set(2, BIWindow::BIDimension(2, 8, 1));

    // 创建一个模拟缓冲区
    // 假设缓冲区大小为 100 字节
    std::vector<uint8_t> buffer(100);
    // 偏移量为0
    size_t               offset = 0;

    BIIterator iterator{num_dims, strides, buffer.data(), offset, window};

    // 打印结果以验证
    for (size_t i = 0; i < num_dims; ++i) {
        std::cout << "Dim " << i << ":\n";
//        std::cout << "  Stride: " << iterator._dims[i]._stride << "\n";
//        std::cout << "  Start: " << iterator._dims[i]._dim_start << "\n";
    }
}

//float MinusOne(float value) {
//    return value - 1.f;
//}
//
//float DoubleRet(float value) {
//    return value * 2;
//}
//
//TEST(test_tranform, transform1) {
//    using namespace BatmanInfer;
//    Tensor<float> f1(2, 3, 4);
//    f1.Rand();
//    f1.Show();
//    f1.Transform(MinusOne);
//    f1.Show();
//}
//
//TEST(test_transform, transform2) {
//    using namespace BatmanInfer;
//    Tensor<float> f2(2, 3, 4);
//    f2.Ones();
//    f2.Show();
//    f2.Transform(DoubleRet);
//    f2.Show();
//}
//
//TEST(test_fill_shape, reshape1) {
//    using namespace BatmanInfer;
//    // Channel, Rows, Cols
//    Tensor<float> f1(2, 3, 4);
//    std::vector<float> values(2 * 3 * 4);
//    // 将1到12填充到values中
//    for (int i = 0; i < 24; ++i)
//        values.at(i) = float(i + 1);
//    f1.Fill(values);
//    f1.Show();
//    f1.Reshape({4, 3, 2});
//    f1.Show();
//}
//
//TEST(test_homework, homework1_flatten1) {
//    using namespace BatmanInfer;
//    Tensor<float> f1(2, 3, 4);
//    LOG(INFO) << "-------------------before Flatten-------------------";
//    f1.Show();
//    f1.Flatten(true);
//    LOG(INFO) << "-------------------after Flatten-------------------";
//    f1.Show();
//    ASSERT_EQ(f1.raw_shapes().size(), 1);
//    ASSERT_EQ(f1.raw_shapes().at(0), 24);
//}
//
//TEST(test_homework, homework1_flatten2) {
//    using namespace BatmanInfer;
//    Tensor<float> f1(12, 24);
//    LOG(INFO) << "-------------------before Flatten-------------------";
//    f1.Show();
//    f1.Flatten(true);
//    LOG(INFO) << "-------------------after Flatten-------------------";
//    f1.Show();
//    ASSERT_EQ(f1.raw_shapes().size(), 1);
//    ASSERT_EQ(f1.raw_shapes().at(0), 24 * 12);
//}
//
//TEST(test_homework, homework2_padding1) {
//    using namespace BatmanInfer;
//    Tensor<float> tensor(3, 4, 5);
//    // channels, rows, cols
//    ASSERT_EQ(tensor.channels(), 3);
//    ASSERT_EQ(tensor.rows(), 4);
//    ASSERT_EQ(tensor.cols(), 5);
//
//    tensor.Fill(1.f);
//    LOG(INFO) << "-------------------before padding-------------------";
//    tensor.Show();
//    tensor.Padding({1, 2, 3, 4}, 0);
//    LOG(INFO) << "-------------------after padding-------------------";
//    tensor.Show();
//    ASSERT_EQ(tensor.rows(), 7);
//    ASSERT_EQ(tensor.cols(), 12);
//
//}
