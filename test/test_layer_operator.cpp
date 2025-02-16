//
// Created by Mason on 2025/1/23.
//

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <runtime/bi_tensor.hpp>
#include <runtime/neon/bi_ne_functions.h>
#include <utils/utils.hpp>

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

void create_input_tensor(BIITensor &tensor, const int hidden_size) {
    std::vector<float16_t> input_data(768 * hidden_size);
    // 初始化输入数据（模拟正态分布）
    for (int i = 0; i < (768 * hidden_size); ++i)
        input_data[i] = static_cast<float16_t>(((i % 32 - 16.0f) / 8.0f));

    auto *src_ptr = reinterpret_cast<float16_t *>(tensor.buffer());
    std::memcpy(src_ptr, input_data.data(), input_data.size() * sizeof(float16_t));
}


TEST(BatmanInferLayer, CPUAttentionTest) {
    // 输入张量
    const BITensorShape input_shape(768,  // sequence
                                    16,
                                    5); // hidden dimension
    const BITensorInfo input_info(input_shape, 1, BIDataType::F16);
    BITensor input;
    input.allocator()->init(input_info);

    // 进行归一化的gamma张量
    const BITensorShape gamma_shape(768);
    const BITensorInfo gamma_info(gamma_shape, 1, BIDataType::F16);
    BITensor gamma;
    gamma.allocator()->init(gamma_info);

    // 权重张量
    const BITensorShape weights_shape(2304,     // input_size (width, 匹配input宽度)
                                      768);    // hidden_units (height)
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
    const BITensorShape output_shape(768,    // hidden_units (width)
                                     16,
                                     5);     // batch_size (height)
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

    PermutationVector perm{0, 2, 1, 3};
    PermutationVector perm2{2, 0, 1, 3};
    PermutationVector perm_final{0, 2, 1, 3};


    // 5. 分配内存
    input.allocator()->allocate();
    weights.allocator()->allocate();
    bias.allocator()->allocate();
    output.allocator()->allocate();
    scalar.allocator()->allocate();
    add_tensor.allocator()->allocate();
    weights2.allocator()->allocate();
    bias2.allocator()->allocate();
    gamma.allocator()->allocate();

    // 模拟数据填充 (实际中应加载量化后的数据)
    // 注意：这里的填充需要符合量化格式
    create_input_tensor(input, 2);
    fill_new_tensor_val(weights, static_cast<float16_t>(1));
    fill_new_tensor_val(bias, static_cast<float16_t>(1));

    fill_new_tensor_val(add_tensor, static_cast<float16_t>(1));
    fill_new_tensor_val(weights2, static_cast<float16_t>(1));
    fill_new_tensor_val(bias2, static_cast<float16_t>(1));
    fill_new_tensor_val(gamma, static_cast<float16_t>(1));

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
                              &gamma,
                              perm,
                              perm2,
                              perm_final,
                              768,
                              16,
                              5,
                              &output);
//    print_new_tensor(input);
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

TEST(BatmanInferLayer, FeedForwardLayerTest) {
    // 输入张量
    const BITensorShape input_shape(768,  // hidden size
                                    16,  // sequence length
                                    5);  // batch size
    const BITensorInfo input_info(input_shape, 1, BIDataType::F16);
    BITensor input;
    input.allocator()->init(input_info);

    const BITensorShape gamma_shape(768);
    const BITensorInfo gamma_info(gamma_shape, 1, BIDataType::F16);
    BITensor gamma;
    gamma.allocator()->init(gamma_info);

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
                                     16,
                                     5);     // batch_size (height)
    const BITensorInfo output_info(output_shape, 1, BIDataType::F16);
    BITensor output;
    output.allocator()->init(output_info);

    const BIActivationLayerInfo act_info(BIActivationFunction::GELU);


    fc_weights.allocator()->allocate();
    fc_bias.allocator()->allocate();
    proj_weights.allocator()->allocate();
    proj_bias.allocator()->allocate();
    output.allocator()->allocate();
    gamma.allocator()->allocate();
    input.allocator()->allocate();
//    // 模拟数据填充 (实际中应加载量化后的数据)
//    // 注意：这里的填充需要符合量化格式
//    fill_new_tensor_val(input, static_cast<float16_t>(1 / 768));
    fill_new_tensor_val(fc_weights, static_cast<float16_t>(1));
    fill_new_tensor_val(fc_bias, static_cast<float16_t>(1));
    fill_new_tensor_val(proj_weights, static_cast<float16_t>(1));
    fill_new_tensor_val(proj_bias, static_cast<float16_t>(1));
    fill_new_tensor_val(gamma, static_cast<float16_t>(1));

    BINEFeedForwardLayer feed_forward_layer;

    feed_forward_layer.configure(&input,
                                 &fc_weights,
                                 &fc_bias,
                                 &proj_weights,
                                 &proj_bias,
                                 &gamma,
                                 act_info,
                                 &output,
                                 5,
                                 16);

    // 获取开始时间点
    auto start = std::chrono::high_resolution_clock::now();


    feed_forward_layer.run();

    // 获取结束时间点
    auto end = std::chrono::high_resolution_clock::now();

    // 计算耗时（以微秒为单位）
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // 输出运行时间
    std::cout << "Function execution time: " << duration.count() << " microseconds" << std::endl;

//    print_new_tensor(output);
}

TEST(BatmanInferLayer, RMSNormTest) {
    // 输入张量
    const BITensorShape input_shape(768,
                                    2,  // hidden size
                                    2); // sequence length
    const BITensorInfo input_info(input_shape, 1, BIDataType::F16);
    BITensor input;
    input.allocator()->init(input_info);

    // gamma张量
    const BITensorShape gamma_shape(768); // hidden size
    const BITensorInfo gamma_info(gamma_shape, 1, BIDataType::F16);
    BITensor gamma;
    gamma.allocator()->init(gamma_info);

    // 输出张量
    const BITensorShape output_shape(768,
                                     2,    // hidden_units (width)
                                     2);     // batch_size (height)
    const BITensorInfo output_info(output_shape, 1, BIDataType::F16);
    BITensor output;
    output.allocator()->init(output_info);

    input.allocator()->allocate();
    gamma.allocator()->allocate();
    output.allocator()->allocate();

    fill_new_tensor_val(gamma, static_cast<float16_t>(1));

    std::vector<float16_t> input_data(768 * 4);
    // 初始化输入数据（模拟正态分布）
    for (int i = 0; i < (768 * 4); ++i)
        input_data[i] = static_cast<float16_t>(((i % 32 - 16.0f) / 8.0f) + (i / 100.0f));

    auto *src_ptr = reinterpret_cast<float16_t *>(input.buffer());
    std::memcpy(src_ptr, input_data.data(), input_data.size() * sizeof(float16_t));

    print_new_tensor(input);


    BINERMSNormLayer rms_norm;
    rms_norm.configure(&input, &gamma, &output);

    // 开始时间节点
    auto start = std::chrono::high_resolution_clock::now();
    rms_norm.run();
    // 结束时间节点
    auto end = std::chrono::high_resolution_clock::now();

    // 计算耗时（以微秒为单位）
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // 输出运行时间
    std::cout << "Function execution time: " << duration.count() << " microseconds" << std::endl;


    print_new_tensor(output);
}

BITensor create_tensor(const BITensorShape &shapes) {
    const BITensorInfo input_info(shapes, 1, BIDataType::F16);
    BITensor input;
    input.allocator()->init(input_info);
    input.allocator()->allocate();
    return input;
}

TEST(BatmanInferLayer, GEMMLayerTest) {
    // 输入张量
    const BITensorShape input_shape(768,
                                    10,  // hidden size
                                    5); // sequence length
    const BITensorShape weight_shape(2304, 768);
    const BITensorShape output_shape(2304, 10, 5);
    const BITensorShape bias_shape(2304);
    auto input = create_tensor(input_shape);
    auto weight = create_tensor(weight_shape);
    auto output = create_tensor(output_shape);
    auto bias = create_tensor(bias_shape);

    fill_new_tensor_val(input, static_cast<float16_t>(1));
    fill_new_tensor_val(weight, static_cast<float16_t>(1));
    fill_new_tensor_val(bias, static_cast<float16_t>(1));

    GEMMInfo gemm_info;
    gemm_info.set_pretranspose_B(false);

    BINEGEMM gemm;
    gemm.configure(&input, &weight, &bias, &output, 1.0f, 1.0f, gemm_info);

    // 开始时间节点
    auto start = std::chrono::high_resolution_clock::now();
    gemm.run();
    // 结束时间节点
    auto end = std::chrono::high_resolution_clock::now();

    // 计算耗时（以微秒为单位）
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // 输出运行时间
    std::cout << "Function execution time: " << duration.count() << " microseconds" << std::endl;

//    print_new_tensor(output);
}

BITensor create_npy_tensor(const std::string &file_name,
                           const BITensorShape &shape) {
    BITensor tensor;
    BITensorInfo tensor_info(shape, 1, BIDataType::F16);
    tensor.allocator()->init(tensor_info);
    tensor.allocator()->allocate();
    utils::read_npy_to_tensor(file_name, tensor);

    return tensor;
}

TEST(BatmanInferLayer, GPT2OneLayerTest) {
    // 先确定需要的算子
    BINEAttentionLayer attention_layer;
    BINEArithmeticAddition add_f;
    BINEFeedForwardLayer feedforward_layer;
    BINEArithmeticAddition add_2_f;

    // 输入张量
    const BITensorShape input_shape(768,  // hidden size
                                    16,  // sequence length
                                    1);  // batch size
    const BITensorShape gamma_shape(768);
    const BITensorShape fc_weights_shape(3072,     // input_size (width, 匹配input宽度)
                                         768);    // hidden_units (height)
    const BITensorShape fc_bias_shape(3072);    // hidden_units
    // 权重张量
    const BITensorShape proj_weights_shape2(768,     // input_size (width, 匹配input宽度)
                                            3072);    // hidden_units (height)
    const BITensorShape proj_bias_shape2(768);    // hidden_units

    const BITensorShape output_shape(768,    // hidden_units (width)
                                     16,
                                     1);     // batch_size (height)
    const BIActivationLayerInfo act_info(BIActivationFunction::GELU);

    // 权重张量
    const BITensorShape weights_shape(2304,     // input_size (width, 匹配input宽度)
                                      768);    // hidden_units (height)

    // 偏置矩阵
    const BITensorShape bias_shape(2304);    // hidden_units

    // 权重张量
    const BITensorShape weights_shape2(768,     // input_size (width, 匹配input宽度)
                                       768);    // hidden_units (height)

    // 偏置矩阵
    const BITensorShape bias_shape2(768);    // hidden_units

    // 标量
    const BITensorShape scalar_shape(1);

    // 相加权重
    const BITensorShape add_shape(16, 16);

    PermutationVector perm{0, 2, 1, 3};
    PermutationVector perm2{2, 0, 1, 3};
    PermutationVector perm_final{0, 2, 1, 3};

    const auto input = create_tensor(input_shape);
    const auto gamma = create_npy_tensor("/Users/mason/Downloads/gpt2_create/rms_attention_1.npy", gamma_shape);
    const auto fc_weights = create_npy_tensor("/Users/mason/Downloads/gpt2_create/mlp_c_fc_weight.npy",
                                              fc_weights_shape);
    const auto fc_bias = create_npy_tensor("/Users/mason/Downloads/gpt2_create/mlp_c_fc_bias.npy", fc_bias_shape);
    const auto proj_weights = create_npy_tensor("/Users/mason/Downloads/gpt2_create/mlp_c_proj_weight.npy",
                                                proj_weights_shape2);
    const auto proj_bias = create_npy_tensor("/Users/mason/Downloads/gpt2_create/mlp_c_proj_bias.npy",
                                             proj_bias_shape2);
    auto output = create_tensor(output_shape);
    const auto weights = create_npy_tensor("/Users/mason/Downloads/gpt2_create/attn_c_attn_weight.npy",
                                           weights_shape);
    const auto bias = create_npy_tensor("/Users/mason/Downloads/gpt2_create/attn_c_attn_bias.npy", bias_shape);
    const auto weights2 = create_npy_tensor("/Users/mason/Downloads/gpt2_create/attn_c_proj_weight_2.npy",
                                            weights_shape2);
    const auto bias2 = create_npy_tensor("/Users/mason/Downloads/gpt2_create/attn_c_proj_bias_2.npy", bias_shape2);
    const auto gamma2 = create_npy_tensor("/Users/mason/Downloads/gpt2_create/mlp_ln_2_weight.npy", gamma_shape);
    const auto scalar = create_tensor(scalar_shape);
    const auto add_tensor = create_npy_tensor("/Users/mason/Downloads/gpt2_create/_attn_Where_output_0.npy", add_shape);

    // 加法结果
    auto add_temp_out = create_tensor(input_shape);
    auto ffn_out = create_tensor(input_shape);
    auto final_out = create_tensor(input_shape);

    fill_new_tensor_val(scalar, static_cast<float16_t>(0.3535533845424652));
    std::vector<float16_t> input_data(768 * 16);
    for (int i = 0; i < 768 * 16; i++) {
        input_data[i] = static_cast<float16_t>(i + 1) / 1000;
    }
    std::memcpy(input.buffer(), input_data.data(), 768 * 16 * sizeof(float16_t));
//    print_new_tensor(input);
//    fill_new_tensor_val(input, static_cast<float16_t>(0.001));

//    fill_new_tensor_val(fc_weights, static_cast<float16_t>(1));
//    fill_new_tensor_val(fc_bias, static_cast<float16_t>(1));
//    fill_new_tensor_val(proj_weights, static_cast<float16_t>(1));
//    fill_new_tensor_val(proj_bias, static_cast<float16_t>(1));
//    fill_new_tensor_val(gamma, static_cast<float16_t>(1));
//    fill_new_tensor_val(weights, static_cast<float16_t>(1));
//    fill_new_tensor_val(bias, static_cast<float16_t>(1));
//
//    fill_new_tensor_val(add_tensor, static_cast<float16_t>(1));
//    fill_new_tensor_val(weights2, static_cast<float16_t>(1));
//    fill_new_tensor_val(bias2, static_cast<float16_t>(1));
//    fill_new_tensor_val(gamma, static_cast<float16_t>(1));


    attention_layer.configure(&input,
                              &weights,
                              &bias,
                              &scalar,
                              &add_tensor,
                              &weights2,
                              &bias2,
                              &gamma,
                              perm,
                              perm2,
                              perm_final,
                              768,
                              16,
                              1,
                              &output);

    add_f.configure(&output, &input, &add_temp_out, BIConvertPolicy::WRAP);

    feedforward_layer.configure(&add_temp_out, &fc_weights,
                                &fc_bias,
                                &proj_weights,
                                &proj_bias,
                                &gamma2,
                                act_info,
                                &ffn_out,
                                1,
                                16);

    add_2_f.configure(&add_temp_out, &ffn_out, &final_out, BIConvertPolicy::WRAP);


    // 开始时间节点
    auto start = std::chrono::high_resolution_clock::now();
    attention_layer.run();
    add_f.run();
    feedforward_layer.run();
    add_2_f.run();
    // 结束时间节点
    auto end = std::chrono::high_resolution_clock::now();

    print_new_tensor(final_out);

    // 计算耗时（以微秒为单位）
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // 输出运行时间
    std::cout << "Function execution time: " << duration.count() << " microseconds" << std::endl;
}