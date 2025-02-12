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

template<typename T>
void fill_new_tensor_val(const BITensor &tensor, const T val) {
    auto tensor_ptr = reinterpret_cast<T *>(tensor.buffer());
    size_t num_elements = tensor.info()->tensor_shape().total_size(); // 获取元素数量
    for (size_t i = 0; i < num_elements; ++i) {
        tensor_ptr[i] = val;
    }
}

/**
 * 根据最小和最大的值, 返回Reasonable quantisation参数来使用float数组
 * @param min
 * @param max
 * @return
 */
BIQuantizationInfo layer_choose_quantization_params(float min,
                                                    float max) {
    // Extend the [min,max] interval to contain 0 so we can represent it exactly
    min = std::min(min, 0.f);
    max = std::max(max, 0.f);

    // Set the quantized min and max in float values
    const float qmin = 0;
    const float qmax = 255;

    // Determine the scale
    const float scale = (max - min) / (qmax - qmin);

    // Determine the zero-point; using affine equation val = (qval-zerop) * scale
    const float zero_point_real = qmin - min / scale;

    // But we need to nudge the zero_point to an integer (exact quantized value)
    std::uint8_t zero_point_nudged = 0;
    if (zero_point_real < qmin)
        zero_point_nudged = qmin;
    else if (zero_point_real > qmax)
        zero_point_nudged = qmax;
    else
        zero_point_nudged = static_cast<std::uint8_t>(support::cpp11::round(zero_point_real));

    BIQuantizationInfo qinfo = BIQuantizationInfo(scale, zero_point_nudged);
    return qinfo;
}

void quantize_values(int size, qasymm8_t *output, float *input, const BIQuantizationInfo &qinfo) {
    for (int i = 0; i < size; i++)
        output[i] = quantize_qasymm8(input[i], qinfo);
    std::cout << "\n";
}


TEST(BatmanInferLayer, CPUAttentionTest) {
    // 输入张量
    const BITensorShape input_shape(16,  // sequence
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
    const BITensorShape output_shape(16,    // hidden_units (width)
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

    PermutationVector perm{2, 0, 1};
    PermutationVector perm2{0, 2, 1};
    PermutationVector perm_final{1, 2, 0};

    // 4. 初始化参数 (使用ACL规范参数)
    const BINormalizationLayerInfo norm_info(
            BINormType::IN_MAP_1D, // 归一化类型
            765,              // norm_size = 特征维度 (D=768)
            1.0f,                 // alpha（缩放因子，对应 γ）
            0.0f,                  // beta（平移项，对应 β）
            0.0f,                 // kappa（禁用）
            true                // is_scaled（自动缩放 alpha）
    );

    // 5. 分配内存
    input.allocator()->allocate();
    weights.allocator()->allocate();
    bias.allocator()->allocate();
    output.allocator()->allocate();
    scalar.allocator()->allocate();
    add_tensor.allocator()->allocate();
    weights2.allocator()->allocate();
    bias2.allocator()->allocate();

    // 模拟数据填充 (实际中应加载量化后的数据)
    // 注意：这里的填充需要符合量化格式
    fill_new_tensor_val(input, static_cast<float16_t>(1));
    fill_new_tensor_val(weights, static_cast<float16_t>(1));
    fill_new_tensor_val(bias, static_cast<float16_t>(1));

    fill_new_tensor_val(add_tensor, static_cast<float16_t>(1));
    fill_new_tensor_val(weights2, static_cast<float16_t>(1));
    fill_new_tensor_val(bias2, static_cast<float16_t>(1));

    auto scalar_ptr = reinterpret_cast<float16_t *>(scalar.buffer());
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
                              norm_info,
                              768,
                              16,
                              &output);

    // 获取开始时间点
    auto start = std::chrono::high_resolution_clock::now();
    attention_layer.run();

    // 获取结束时间点
    auto end = std::chrono::high_resolution_clock::now();

    // 计算耗时（以微秒为单位）
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // 输出运行时间
    std::cout << "Function execution time: " << duration.count() << " milliseconds" << std::endl;

//    print_new_tensor(output);
}

TEST(BatmanInferLayer, FeedForwardLayerTest) {
    // 输入张量
    const BITensorShape input_shape(768,  // hidden size
                                    16); // sequence length
    const BITensorInfo input_info(input_shape, 1, BIDataType::F16);
    BITensor input;
    input.allocator()->init(input_info);

    // 权重张量
    const BITensorShape fc_weights_shape(3072,     // input_size (width, 匹配input宽度)
                                         768);    // hidden_units (height)
    const BITensorInfo fc_weights_info(fc_weights_shape, 1, BIDataType::F16);
    BITensor fc_weights;
    fc_weights.allocator()->init(fc_weights_info);

    // 偏置矩阵
    const BITensorShape fc_bias_shape(3072);    // hidden_units
    const BITensorInfo fc_bias_info(fc_bias_shape, 1, BIDataType::F16);
    BITensor fc_bias;
    fc_bias.allocator()->init(fc_bias_info);

    // 权重张量
    const BITensorShape proj_weights_shape2(768,     // input_size (width, 匹配input宽度)
                                            3072);    // hidden_units (height)
    const BITensorInfo proj_weights_info2(proj_weights_shape2, 1, BIDataType::F16);
    BITensor proj_weights;
    proj_weights.allocator()->init(proj_weights_info2);

    // 偏置矩阵
    const BITensorShape proj_bias_shape2(768);    // hidden_units
    const BITensorInfo proj_bias_info2(proj_bias_shape2, 1, BIDataType::F16);
    BITensor proj_bias;
    proj_bias.allocator()->init(proj_bias_info2);

    // 输出张量
    const BITensorShape output_shape(768,    // hidden_units (width)
                                     16);     // batch_size (height)
    const BITensorInfo output_info(output_shape, 1, BIDataType::F16);
    BITensor output;
    output.allocator()->init(output_info);

    // 4. 初始化参数 (使用ACL规范参数)
    const BINormalizationLayerInfo norm_info(
            BINormType::CROSS_MAP, // 归一化类型
            5,                                // 归一化窗口大小
            0.0001f,                          // epsilon
            0.75f,                            // beta
            1.0f,                             // kappa
            false                             // 是否跨通道
    );

    const BIActivationLayerInfo act_info(BIActivationFunction::GELU);


    fc_weights.allocator()->allocate();
    fc_bias.allocator()->allocate();
    proj_weights.allocator()->allocate();
    proj_bias.allocator()->allocate();
    output.allocator()->allocate();
//    // 模拟数据填充 (实际中应加载量化后的数据)
//    // 注意：这里的填充需要符合量化格式
//    fill_new_tensor_val(input, static_cast<float16_t>(1 / 768));
    fill_new_tensor_val(fc_weights, static_cast<float16_t>(1));
    fill_new_tensor_val(fc_bias, static_cast<float16_t>(1));
    fill_new_tensor_val(proj_weights, static_cast<float16_t>(1));
    fill_new_tensor_val(proj_bias, static_cast<float16_t>(1));
    // 获取开始时间点
    auto start = std::chrono::high_resolution_clock::now();
    BINEFeedForwardLayer feed_forward_layer;
    feed_forward_layer.configure(&input,
                                 &fc_weights,
                                 &fc_bias,
                                 &proj_weights,
                                 &proj_bias,
                                 act_info,
                                 norm_info,
                                 &output);

    input.allocator()->info().set_tensor_shape(BITensorShape(768, 1));
    output.allocator()->info().set_tensor_shape(BITensorShape(768, 1));

    // 5. 分配内存
    input.allocator()->allocate();

//    // 获取开始时间点
//    auto start = std::chrono::high_resolution_clock::now();
    feed_forward_layer.run();

    input.allocator()->info().set_tensor_shape(BITensorShape(768, 2));
    output.allocator()->info().set_tensor_shape(BITensorShape(768, 2));

    fill_new_tensor_val(input, static_cast<float16_t>(1 / 768));

    feed_forward_layer.run();

    input.allocator()->info().set_tensor_shape(BITensorShape(768, 3));
    output.allocator()->info().set_tensor_shape(BITensorShape(768, 3));

    feed_forward_layer.run();

    input.allocator()->info().set_tensor_shape(BITensorShape(768, 4));
    output.allocator()->info().set_tensor_shape(BITensorShape(768, 4));

    feed_forward_layer.run();

    // 获取结束时间点
    auto end = std::chrono::high_resolution_clock::now();

    // 计算耗时（以微秒为单位）
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // 输出运行时间
    std::cout << "Function execution time: " << duration.count() << " milliseconds" << std::endl;

    print_new_tensor(output);
}

TEST(BatmanInferLayer, RMSNormTest) {
    // 输入张量
    const BITensorShape input_shape(768,  // hidden size
                                    16); // sequence length
    const BITensorInfo input_info(input_shape, 1, BIDataType::F16);
    BITensor input;
    input.allocator()->init(input_info);

    // gamma张量
    const BITensorShape gamma_shape(768); // hidden size
    const BITensorInfo gamma_info(gamma_shape, 1, BIDataType::F16);
    BITensor gamma;
    gamma.allocator()->init(gamma_info);

    // 输出张量
    const BITensorShape output_shape(768,    // hidden_units (width)
                                     16);     // batch_size (height)
    const BITensorInfo output_info(output_shape, 1, BIDataType::F16);
    BITensor output;
    output.allocator()->init(output_info);

    input.allocator()->allocate();
    gamma.allocator()->allocate();
    output.allocator()->allocate();

    fill_new_tensor_val(gamma, static_cast<float16_t>(1));

    std::vector<float16_t> input_data(768 * 16);
    // 初始化输入数据（模拟正态分布）
    for (int i = 0; i < (768 * 16); ++i)
        input_data[i] = static_cast<float16_t>((i % 32 - 16.0f) / 8.0f);

    auto *src_ptr = reinterpret_cast<float16_t *>(input.buffer());
    std::memcpy(src_ptr, input_data.data(), input_data.size() * sizeof(float16_t));


    BINERMSNormLayer rms_norm;
    rms_norm.configure(&input, &gamma, &output);

    rms_norm.run();


    print_new_tensor(output);
}