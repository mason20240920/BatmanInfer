//
// Created by Mason on 2025/1/23.
//

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <runtime/bi_tensor.hpp>
#include <runtime/neon/bi_ne_functions.h>

using namespace BatmanInfer;


TEST(BatmanInferLayer, RNNLayerTest) {
    // 输入张量
    const BITensorShape input_shape(64,     // input_size (width)
                                    32);     // batch_size (height)
    const BITensorInfo input_info(input_shape, 1, BIDataType::F32);
    BITensor input;
    input.allocator()->init(input_info);

    // 权重张量
    const BITensorShape weights_shape(64,     // input_size (width, 匹配input宽度)
                                      128);    // hidden_units (height)
    const BITensorInfo weights_info(weights_shape, 1, BIDataType::F32);
    BITensor weights;
    weights.allocator()->init(weights_info);

    // 循环权重张量
    const BITensorShape recurrent_weights_shape(128,    // hidden_units (width)
                                                128);    // hidden_units (height)
    const BITensorInfo recurrent_weights_info(recurrent_weights_shape, 1, BIDataType::F32);
    BITensor recurrent_weights;
    recurrent_weights.allocator()->init(recurrent_weights_info);

    // 偏置矩阵
    const BITensorShape bias_shape(128);    // hidden_units
    const BITensorInfo bias_info(bias_shape, 1, BIDataType::F32);
    BITensor bias;
    bias.allocator()->init(bias_info);

    // 隐藏层张量
    const BITensorShape hidden_state_shape(128,    // hidden_units (width)
                                           32);     // batch_size (height, 匹配input高度)
    const BITensorInfo hidden_state_info(hidden_state_shape, 1, BIDataType::F32);
    BITensor hidden_state;
    hidden_state.allocator()->init(hidden_state_info);

    // 输出张量
    const BITensorShape output_shape(128,    // hidden_units (width)
                                     32);     // batch_size (height)
    const BITensorInfo output_info(output_shape, 1, BIDataType::F32);
    BITensor output;
    output.allocator()->init(output_info);

    // 5. 分配内存
    input.allocator()->allocate();
    weights.allocator()->allocate();
    recurrent_weights.allocator()->allocate();
    bias.allocator()->allocate();
    hidden_state.allocator()->allocate();
    output.allocator()->allocate();

    // 7. 创建RNN层配置
    BINERNNLayer rnn_layer;
    BIActivationLayerInfo activation_info(BIActivationLayerInfo::ActivationFunction::TANH); // 激活函数使用tanh

    // 8. 配置RNN层
    rnn_layer.configure(&input, &weights, &recurrent_weights, &bias, &hidden_state, &output, activation_info);

    auto start = std::chrono::high_resolution_clock::now();

    // 10. 运行RNN层
    rnn_layer.run();

    // 记录结束时间点
    auto end = std::chrono::high_resolution_clock::now();

    // 计算时间差
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // 打印结果
    std::cout << "Function execution time: " << duration.count() << " microseconds" << std::endl;
}

void print_new_tensor(const BITensor &tensor) {
    BIIOFormatInfo format;
    format.element_delim = ", ";  // 元素之间用逗号分隔
    format.row_delim = "\n";      // 每行换行
    format.align_columns = 1;     // 对齐列

    tensor.print(std::cout, format);
}

void fill_new_tensor_val(const BITensor &tensor, const float16_t val) {
    auto tensor_ptr = reinterpret_cast<float16_t *>(tensor.buffer());
    size_t num_elements = tensor.info()->tensor_shape().total_size(); // 获取元素数量
    for (size_t i = 0; i < num_elements; ++i) {
        tensor_ptr[i] = val;
    }
}


TEST(BatmanInferLayer, CPUAttentionTest) {
    // 输入张量
    const BITensorShape input_shape(1,   // batch size
                                    16,  // sequence
                                    768); // hidden dimension
    const BITensorInfo input_info(input_shape, 1, BIDataType::F16);
    BITensor input;
    input.allocator()->init(input_info);

    // 权重张量
    const BITensorShape weights_shape(768,     // input_size (width, 匹配input宽度)
                                      2304);    // hidden_units (height)
    const BITensorInfo weights_info(weights_shape, 1, BIDataType::F16);
    BITensor weights;
    weights.allocator()->init(weights_info);

    // 偏置矩阵
    const BITensorShape bias_shape(2304);    // hidden_units
    const BITensorInfo bias_info(bias_shape, 1, BIDataType::F16);
    BITensor bias;
    bias.allocator()->init(bias_info);

    // 权重张量
    const BITensorShape weights_shape2(768,     // input_size (width, 匹配input宽度)
                                       768);    // hidden_units (height)
    const BITensorInfo weights_info2(weights_shape2, 1, BIDataType::F16);
    BITensor weights2;
    weights2.allocator()->init(weights_info2);

    // 偏置矩阵
    const BITensorShape bias_shape2(768);    // hidden_units
    const BITensorInfo bias_info2(bias_shape2, 1, BIDataType::F16);
    BITensor bias2;
    bias2.allocator()->init(bias_info2);

    // 输出张量
    const BITensorShape output_shape(1,
                                     16,    // hidden_units (width)
                                     768);     // batch_size (height)
    const BITensorInfo output_info(output_shape, 1, BIDataType::F16);
    BITensor output;
    output.allocator()->init(output_info);

    // 标量
    const BITensorShape scalar_shape(1);
    const BITensorInfo scalar_info(scalar_shape, 1, BIDataType::F16);
    BITensor scalar;
    scalar.allocator()->init(scalar_info);

    // 相加权重
    const BITensorShape add_shape(16, 16);
    const BITensorInfo add_info(add_shape, 1, BIDataType::F16);
    BITensor add_tensor;
    add_tensor.allocator()->init(add_info);

    PermutationVector perm{3, 1, 2, 0};
    PermutationVector perm2{1, 3, 2, 0};
    PermutationVector perm_final{1, 2, 0};

    // 5. 分配内存
    input.allocator()->allocate();
    weights.allocator()->allocate();
    bias.allocator()->allocate();
    output.allocator()->allocate();
    scalar.allocator()->allocate();
    add_tensor.allocator()->allocate();
    weights2.allocator()->allocate();
    bias.allocator()->allocate();

    // 模拟数据填充 (实际中应加载量化后的数据)
    // 注意：这里的填充需要符合量化格式
    fill_new_tensor_val(input, 1 / 768);
    fill_new_tensor_val(weights, 1);
    fill_new_tensor_val(bias, 1);

    fill_new_tensor_val(add_tensor, 1);
    fill_new_tensor_val(weights2, 1);
//    fill_new_tensor_val(bias2, 1);

    auto scalar_ptr = reinterpret_cast<float *>(scalar.buffer());
    scalar_ptr[0] = 0.5f;

    BINEAttentionLayer attention_layer;
    attention_layer.configure(&input,
                              &weights,
                              &bias,
                              &scalar,
                              &add_tensor,
                              &weights2,
                              &bias2,
                              perm,
                              perm2,
                              perm_final,
                              &output);

    // 获取开始时间点
    auto start = std::chrono::high_resolution_clock::now();
    attention_layer.run();

    // 获取结束时间点
    auto end = std::chrono::high_resolution_clock::now();

    // 计算耗时（以微秒为单位）
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // 输出运行时间
    std::cout << "Function execution time: " << duration.count() << " microseconds" << std::endl;

//    print_new_tensor(output);
}