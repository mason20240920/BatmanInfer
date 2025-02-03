//
// Created by Mason on 2025/1/23.
//

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <runtime/bi_tensor.hpp>
#include <runtime/neon/functions/bi_ne_rnn_layer.hpp>
#include <runtime/neon/functions/bi_ne_attention_layer.hpp>

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

TEST(BatmanInferLayer, CPUAttentionTest) {
    // 输入张量
    const BITensorShape input_shape(1,   // batch size
                                    16,  // sequence
                                    768); // hidden dimension
    const BITensorInfo input_info(input_shape, 1, BIDataType::F32);
    BITensor input;
    input.allocator()->init(input_info);

    // 权重张量
    const BITensorShape weights_shape(768,     // input_size (width, 匹配input宽度)
                                      2304);    // hidden_units (height)
    const BITensorInfo weights_info(weights_shape, 1, BIDataType::F32);
    BITensor weights;
    weights.allocator()->init(weights_info);

    // 偏置矩阵
    const BITensorShape bias_shape(2304);    // hidden_units
    const BITensorInfo bias_info(bias_shape, 1, BIDataType::F32);
    BITensor bias;
    bias.allocator()->init(bias_info);

    // 输出张量
    const BITensorShape output_shape(16,
                                     16,    // hidden_units (width)
                                     12,
                                     1);     // batch_size (height)
    const BITensorInfo output_info(output_shape, 1, BIDataType::F32);
    BITensor output;
    output.allocator()->init(output_info);

    // 标量
    const BITensorShape scalar_shape(1);
    const BITensorInfo scalar_info(scalar_shape, 1, BIDataType::F32);
    BITensor scalar;
    scalar.allocator()->init(scalar_info);

    PermutationVector perm{3, 1, 2, 0};
    PermutationVector perm2{1, 3, 2, 0};

    // 5. 分配内存
    input.allocator()->allocate();
    weights.allocator()->allocate();
    bias.allocator()->allocate();
    output.allocator()->allocate();
    scalar.allocator()->allocate();

    // 模拟数据填充 (实际中应加载量化后的数据)
    // 注意：这里的填充需要符合量化格式
    auto input_ptr = reinterpret_cast<float *>(input.buffer());
    for (size_t i = 0; i < input.info()->total_size(); ++i) {
        input_ptr[i] = 1 / 768; // 假设输入数据全为 zero_point
    }

    auto weights_ptr = reinterpret_cast<float *>(weights.buffer());
    size_t num_elements = weights.info()->tensor_shape().total_size(); // 获取元素数量
    for (size_t i = 0; i < num_elements; ++i) {
        weights_ptr[i] = 1.0f; // 假设权重数据全为 zero_point
    }

    auto biases_ptr = reinterpret_cast<float *>(bias.buffer());
    for (size_t i = 0; i < bias.info()->total_size() / sizeof(int32_t); ++i) {
        biases_ptr[i] = 1; // 偏置为零
    }

    auto scalar_ptr = reinterpret_cast<float *>(scalar.buffer());
    scalar_ptr[0] = 0.5f;

    BINEAttentionLayer attention_layer;
    attention_layer.configure(&input, &weights, &bias, &scalar, perm, perm2, &output);

    attention_layer.run();

    BIIOFormatInfo format;
    format.element_delim = ", ";  // 元素之间用逗号分隔
    format.row_delim = "\n";      // 每行换行
    format.align_columns = 1;     // 对齐列

    output.print(std::cout, format);
}