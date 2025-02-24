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

    int batch_size = 1;
    int sequence_len = 16;
    int kv_one_len = 1;
    int head_num = 1;
    int head_dim = 2;
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
//    settings.fixed_format(true);

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
        a_ptr[i] = static_cast<float16_t>(i); // 示例数据
    }
    for (int i = 0; i < shape_b.total_size(); ++i) {
        b_ptr[i] = static_cast<float16_t>(1); // 示例数据
    }

    // 开始时间节点
//    auto start = std::chrono::high_resolution_clock::now();
    matmul.run();

    tensor_c.print(std::cout, format);

    // 更新数据
    for (int i = 0; i < shape_a.total_size(); ++i) {
        a_ptr[i] = static_cast<float16_t>(i + 1); // 示例数据
    }

    std::cout << "=======" << std::endl;

    matmul.run();
    tensor_c.print(std::cout, format);
}

void match_info(BITensor &tensor, const std::vector<float16_t> &vec) {
    auto shape = tensor.info()->tensor_shape().total_size();
    auto tensor_data = reinterpret_cast<float16_t *>(tensor.buffer());
    for (int i = 0; i < shape; i++) {
        tensor_data[i] = vec[i];
    }
}

TEST(KVCaches, DynamicGemm) {
    BIScheduler::get().set_num_threads(1);
    // 进行矩阵计算的KVCaches
    BIIOFormatInfo format;
    format.element_delim = ", ";  // 元素之间用逗号分隔
    format.row_delim = "\n";      // 每行换行
    format.align_columns = true;     // 对齐列
    int batch_size = 1;
    int sequence_len = 2;
    // 测试动态输入NEGEMM的过程
    BITensorShape tensor_a_shape(2, sequence_len, batch_size);
    BITensorShape tensor_b_shape(4, 2);
    BITensorShape tensor_bias_shape(4, sequence_len);
    BITensorShape tensor_d_shape(4, sequence_len, batch_size);

    BITensorInfo tensor_a_info(tensor_a_shape, 1, BIDataType::F16);
    BITensorInfo tensor_b_info(tensor_b_shape, 1, BIDataType::F16);
    BITensorInfo tensor_bias_info(tensor_bias_shape, 1, BIDataType::F16);
    BITensorInfo tensor_d_info(tensor_d_shape, 1, BIDataType::F16);

    BITensor tensor_a, tensor_b, bias, tensor_d;

    // 初始化
    tensor_a.allocator()->init(tensor_a_info);
    tensor_b.allocator()->init(tensor_b_info);
    bias.allocator()->init(tensor_bias_info);
    tensor_d.allocator()->init(tensor_d_info);

    tensor_a.allocator()->allocate();
    tensor_b.allocator()->allocate();
    bias.allocator()->allocate();
    tensor_d.allocator()->allocate();

    // 进行赋值
    std::vector<float16_t> data_a{1, 2, 3, 4};
    std::vector<float16_t> data_b{1, 1, 1, 1, 1, 1, 1, 1};
    std::vector<float16_t> data_bias{3, 3, 3, 3};
    match_info(tensor_a, data_a);
    match_info(tensor_b, data_b);
    match_info(bias, data_bias);



    // 运行推理
    BINEGEMM gemm;
    GEMMInfo gemm_info;
    gemm_info.set_fast_math(true);

    gemm.configure(&tensor_a, &tensor_b, &bias, &tensor_d, 1.0f, 1.0f, gemm_info);

    gemm.run();

    tensor_d.print(std::cout, format);

    // 动态更新
    sequence_len = 3;
    batch_size = 2;
    tensor_a_shape = BITensorShape(2, sequence_len, batch_size);
    tensor_d_shape = BITensorShape(4, sequence_len, batch_size);

    tensor_a_info = BITensorInfo(tensor_a_shape, 1, BIDataType::F16);
    tensor_d_info = BITensorInfo(tensor_d_shape, 1, BIDataType::F16);

    // 初始化
    tensor_a.allocator()->init(tensor_a_info);
    tensor_d.allocator()->init(tensor_d_info);

    tensor_a.allocator()->allocate();
    tensor_d.allocator()->allocate();

    data_a = {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    match_info(tensor_a, data_a);
    gemm.run();

    tensor_d.print(std::cout, format);
}