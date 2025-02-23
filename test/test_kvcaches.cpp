//
// Created by Mason on 2025/2/21.
//

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <runtime/neon/bi_ne_functions.h>
#include <runtime/bi_tensor.hpp>
#include <utils/utils.hpp>
#include "function_info/bi_MatMulInfo.h"
#include "runtime/bi_scheduler.hpp"
#include <thread>

using namespace BatmanInfer;

TEST(KVCaches, NEGEMMCaches) {
    BIScheduler::get().set_num_threads(std::thread::hardware_concurrency());
    // 进行矩阵计算的KVCaches
    BIIOFormatInfo format;
    format.element_delim = ", ";  // 元素之间用逗号分隔
    format.row_delim = "\n";      // 每行换行
    format.align_columns = true;     // 对齐列

    int batch_size = 5;
    int sequence_len = 16;
    int kv_one_len = 16;
    int head_num = 12;
    int head_dim = 64;
    // 定义输入和输出张量的形状
    BITensorShape shape_a(head_dim, kv_one_len, head_num, batch_size); // 左矩阵 (3x2)
    BITensorShape shape_b(sequence_len, head_dim, head_num, batch_size); // 右矩阵 (4x2)，需要转置为 (2x4)
    BITensorShape shape_c(sequence_len, kv_one_len, head_num, batch_size); // 输出矩阵 (4x3)

    // 创建输入和输出张量
    BITensor tensor_a, tensor_b, tensor_c;

    // 配置张量
    tensor_a.allocator()->init(BITensorInfo(shape_a, 1, BIDataType::F16));
    tensor_b.allocator()->init(BITensorInfo(shape_b, 1, BIDataType::F16));
    tensor_c.allocator()->init(BITensorInfo(shape_c, 1, BIDataType::F16));

    tensor_a.info()->set_are_values_constant(false);
    tensor_b.info()->set_are_values_constant(false);
    // 定义 MatMul 配置信息
    BIMatMulInfo matmul_info; // 不转置左矩阵，转置右矩阵
    matmul_info.adj_lhs(false).adj_rhs(false);
    BICpuMatMulSettings settings;
    settings.fast_math(true); // 启用快速数学模式
    settings.fixed_format(true);

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
    auto a_ptr = reinterpret_cast<float16_t *>(tensor_a.buffer());
    auto b_ptr = reinterpret_cast<float16_t *>(tensor_b.buffer());
    for (int i = 0; i < shape_a.total_size(); ++i) {
        a_ptr[i] = static_cast<float16_t>(i * 0.01); // 示例数据
    }
    for (int i = 0; i < shape_b.total_size(); ++i) {
        b_ptr[i] = static_cast<float16_t>(1 * 0.1); // 示例数据
    }

    // 开始时间节点
    auto start = std::chrono::high_resolution_clock::now();
    matmul.run();
    // 结束时间节点
    auto end = std::chrono::high_resolution_clock::now();
    // 计算耗时（以微秒为单位）
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // 输出运行时间
    std::cout << "Function execution time: " << duration.count() << " microseconds" << std::endl;
}