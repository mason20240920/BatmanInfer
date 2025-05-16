//
// Created by Mason on 2025/4/10.
//
#include <thread>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <runtime/neon/bi_ne_functions.h>
#include <runtime/bi_tensor.hpp>

#include "runtime/bi_scheduler.hpp"
#include "runtime/neon/functions/BINEReductionOperation.hpp"
#include "utils/utils.hpp"

namespace MemAllocTest {
    using namespace BatmanInfer;

    void print_tensor(const BatmanInfer::BITensor &tensor,
                      const std::string &name = "temp",
                      const BatmanInfer::BIIOFormatInfo::PrintRegion region =
                              BatmanInfer::BIIOFormatInfo::PrintRegion::Full) {
        std::cout << name << std::endl;
        BatmanInfer::BIIOFormatInfo format;
        format.element_delim = ", "; // 元素之间用逗号分隔
        format.row_delim = "\n"; // 每行换行
        format.align_columns = true; // 对齐列
        format.print_region = region;

        tensor.print(std::cout, format);
    }

    BITensor create_norm_input(std::vector<int> shapes, const std::string &file_path = "") {
        BITensorShape input_shape;
        if (shapes.size() == 3)
            input_shape = BITensorShape(shapes[2], shapes[1], shapes[0]); // [M, K]
        else if (shapes.size() == 2)
            input_shape = BITensorShape(shapes[1], shapes[0]);
        else if (shapes.size() == 1)
            input_shape = BITensorShape(shapes[0]);
        else
            input_shape = BITensorShape(shapes[3], shapes[2], shapes[1], shapes[0]); // [M, K]
        BITensor input = utils::create_type_tensor(file_path,
                                                   input_shape,
                                                   BIDataType::F16);
        return input;
    }

    BITensor create_norm_bias(const int &bias_dim, const std::string &file_path = "") {
        BITensorShape input_shape = BITensorShape(bias_dim);
        BITensor input = utils::create_type_tensor(file_path,
                                                   input_shape,
                                                   BIDataType::S32);
        return input;
    }

    BITensor create_qasymm8(float input_scale,
                            int zero_point,
                            std::vector<int> shapes,
                            BIQuantizationInfo &q_info,
                            const std::string &file_path = "") {
        q_info = BIQuantizationInfo(input_scale, zero_point);
        BITensorShape input_shape;
        if (shapes.size() == 3)
            input_shape = BITensorShape(shapes[2], shapes[1], shapes[0]); // [M, K]
        else
            input_shape = BITensorShape(shapes[3], shapes[2], shapes[1], shapes[0]); // [M, K]
        BITensor input = utils::create_type_tensor(file_path, input_shape,
                                                   BIDataType::QASYMM8_SIGNED);
        input.info()->set_quantization_info(q_info);
        return input;
    }

    BITensor create_per_channel(const std::vector<float> &weight_scales,
                                std::vector<int> shapes,
                                BIQuantizationInfo &weights_qinfo,
                                const std::string &file_path = "") {
        weights_qinfo = BIQuantizationInfo(weight_scales);
        BITensorShape weights_shape(shapes[1], shapes[0]); // [K, N]
        BITensor weights = utils::create_type_tensor(file_path,
                                                     weights_shape,
                                                     BIDataType::QSYMM8_PER_CHANNEL);
        weights.info()->set_quantization_info(weights_qinfo);
        return weights;
    }

    template<typename T>
    void fill_tensor_val_with_arr(const BatmanInfer::BITensor &tensor, const std::vector<T> val) {
        auto tensor_ptr = reinterpret_cast<T *>(tensor.buffer());
        const size_t num_elements = tensor.info()->tensor_shape().total_size(); // 获取元素数量
        for (size_t i = 0; i < num_elements; i++) {
            tensor_ptr[i] = val[i];
        }
    } /**
 * 性能统计结构体
 */
    struct PerfStats {
        double avg_ms; // 平均耗时
        double std_dev_ms; // 标准差
        double min_ms; // 最小耗时
        double max_ms; // 最大耗时
        size_t iterations; // 有效迭代次数
    };
}

TEST(MemAlloc, TensorAlloc) {
    using namespace BatmanInfer;
    BITensor tensor;
    tensor.allocator()->init(BITensorInfo(BITensorShape(4, 4), 1, BIDataType::F16));
    tensor.allocator()->allocate();

    BITensorShape sub_shape(3, 3); // 要提取 64x64 的子区域
    BITensorInfo sub_info(sub_shape, 1, BIDataType::F16);
    sub_info.set_format(Format::F16);

    auto input_ptr = reinterpret_cast<float16_t *>(tensor.buffer());
    for (int i = 0; i < 256; i++) {
        input_ptr[i] = static_cast<float16_t>(i);
    }
    MemAllocTest::print_tensor(tensor, "input");

    // 4. 创建子张量
    BITensor sub_tensor;
    sub_tensor.allocator()->init(*tensor.allocator(), sub_info);
    BIIOFormatInfo format;
    format.element_delim = ", "; // 元素之间用逗号分隔
    format.row_delim = "\n"; // 每行换行
    format.align_columns = 1; // 对齐列

    sub_tensor.print(std::cout, format);
}

TEST(MemAllocGPT2, GPTAllocDynamic) {
    using namespace BatmanInfer;
    // BIScheduler::set(BIScheduler::Type::OMP);
    BIScheduler::get().set_num_threads(std::thread::hardware_concurrency());
    BIMemoryGroup group{BIMemoryManagerOnDemand::make_default()};

    int batch_size = 1;
    int seq_len = 1;
    // 1. 初始化一个最大input算子
    BITensor original_input_tensor;
    BITensorShape original_input_tensor_shape(16, 20);
    original_input_tensor.allocator()->init(BITensorInfo(original_input_tensor_shape, 1, BIDataType::U32));
    original_input_tensor.allocator()->allocate();

    // 1.1 初始化一个小型算子
    BITensor input_tensor;
    BITensorShape input_tensor_shape(seq_len, batch_size);
    BITensorInfo input_info(input_tensor_shape, 1, BIDataType::U32);
    input_info.set_format(Format::U32);
    input_tensor.allocator()->init(*original_input_tensor.allocator(), input_info);
    std::vector<uint32_t> indices_data{0};
    MemAllocTest::fill_tensor_val_with_arr(input_tensor, indices_data);


    // 2. Gather的权重
    BITensorShape gather_weight_shape(768, 6003);
    const std::string &weight_path =
            "./input_res/transformer_wte_weight.npy";
    BITensor weight = utils::create_type_tensor(
        weight_path, gather_weight_shape,
        BIDataType::F16);
    // MemAllocTest::print_tensor(weight, "weight");

    // 3. 输出原始矩阵
    BITensor original_gather_output_tensor;
    BITensorShape original_gather_output_tensor_shape(768, 16, 20);
    original_gather_output_tensor.allocator()->init(
        BITensorInfo(original_gather_output_tensor_shape, 1, BIDataType::F16));
    original_gather_output_tensor.allocator()->allocate();

    BITensor original_attn_rms_output_tensor;
    original_attn_rms_output_tensor.allocator()->init(
        BITensorInfo(original_gather_output_tensor_shape, 1, BIDataType::F16));
    original_attn_rms_output_tensor.allocator()->allocate();

    // 3.1 输出矩阵的子矩阵
    BITensor gather_output_tensor;
    BITensorShape gather_output_tensor_shape(768, seq_len, batch_size);
    BITensorInfo gather_output_info(gather_output_tensor_shape, 1, BIDataType::F16);
    gather_output_info.set_format(Format::F16);
    gather_output_tensor.allocator()->init(*original_gather_output_tensor.allocator(), gather_output_info);


    // 2. 进行NEGather筛选
    BINEGather gather_layer;
    gather_layer.configure(&weight, &input_tensor, &gather_output_tensor, 1);
    gather_layer.run();

    // 3. Add权重的获取
    BITensorShape add_wte_weight_shape(768, 16);
    const std::string &add_wte_weight_path = "./input_res/add_wte_weights.npy";
    BITensor add_wte_weight = utils::create_type_tensor(
        add_wte_weight_path, add_wte_weight_shape,
        BIDataType::F16);

    // 临时的数据
    BITensor sub_add_weight;
    BITensorShape sub_add_weight_shape(768, seq_len);
    BITensorInfo sub_add_weight_info(sub_add_weight_shape, 1, BIDataType::F16);
    sub_add_weight_info.set_format(Format::F16);
    sub_add_weight.allocator()->init(*add_wte_weight.allocator(), sub_add_weight_info);
    // 4. Add输出的原始最大值
    BITensor original_add_output_tensor;
    original_add_output_tensor.allocator()->init(
        BITensorInfo(original_gather_output_tensor_shape, 1, BIDataType::F16));
    original_add_output_tensor.allocator()->allocate();
    // 4.1 Add输出的新的数据格式
    BITensor add_output_tensor;
    add_output_tensor.allocator()->init(*original_add_output_tensor.allocator(), gather_output_info);
    BINEArithmeticAddition add_layer;
    add_layer.configure(&gather_output_tensor, &sub_add_weight, &add_output_tensor, BIConvertPolicy::SATURATE);
    add_layer.run();
    // 5 获取Attention模块的权重
    // 5.1 gamma权重
    BITensorShape attn_gamma_weights_shape(768);
    const std::string &gamma_weights_path = "./input_res/attn_gamma_weights.npy";
    BITensor attn_gamma_weights = utils::create_type_tensor(
        gamma_weights_path, attn_gamma_weights_shape,
        BIDataType::F16);
    BITensor attn_origin_o_tensor;
    attn_origin_o_tensor.allocator()->init(BITensorInfo(original_gather_output_tensor_shape, 1, BIDataType::F16));
    attn_origin_o_tensor.allocator()->allocate();
    BITensor attn_output_tensor;
    attn_output_tensor.allocator()->init(*attn_origin_o_tensor.allocator(), gather_output_info);
    constexpr float attn_gemm_i_scale = 0.006409900328692268f;
    constexpr int attn_gemm_i_zero = -6;
    // 5.2 c_attn权重和偏置值
    BITensorShape attn_qkv_weights_shape(2304, 768);
    const std::string &attn_qkv_weights_path = "./input_res/c_attn_weights.npy";
    BITensor attn_qkv_weights = utils::create_type_tensor(attn_qkv_weights_path, attn_qkv_weights_shape,
                                                          BIDataType::QSYMM8_PER_CHANNEL);
    std::ifstream attn_qkv_weights_scale_file(
        "./input_res/c_attn_scales.txt");
    float value;
    std::vector<float> attn_qkv_weights_scales;
    while (attn_qkv_weights_scale_file >> value) {
        attn_qkv_weights_scales.push_back(value);
    }
    BIQuantizationInfo attn_qkv_weights_qinfo(attn_qkv_weights_scales);
    attn_qkv_weights.info()->set_quantization_info(attn_qkv_weights_qinfo);
    BITensorShape attn_qkv_bias_shape(2304);
    const std::string &attn_qkv_bias_path = "./input_res/c_attn_bias.npy";
    BITensor attn_qkv_bias = utils::create_type_tensor(attn_qkv_bias_path, attn_qkv_bias_shape,
                                                       BIDataType::S32);
    BITensorShape attn_c_proj_weights_shape(768, 768);
    const std::string &attn_c_proj_weights_path = "./input_res/p_attn_weights.npy";
    BITensor attn_c_proj_weights = utils::create_type_tensor(attn_c_proj_weights_path, attn_c_proj_weights_shape,
                                                             BIDataType::F16);
    BITensorShape attn_c_proj_bias_shape(768);
    const std::string &attn_c_proj_bias_path = "./input_res/p_attn_bias.npy";
    BITensor attn_c_proj_bias = utils::create_type_tensor(attn_c_proj_bias_path, attn_c_proj_bias_shape,
                                                          BIDataType::F16);
    // MemAllocTest::print_tensor(attn_c_proj_weights, "attn_c_proj_weights");
    constexpr float attn_gemm_o_scale = 0.08648063435274012f;
    constexpr int attn_gemm_o_zero = -9;
    constexpr float query_scale = 0.04602363623824774f;
    constexpr int query_zp = -11;
    constexpr float value_scale = 0.08648063435274012f;
    constexpr int value_zp = -9;
    constexpr float key_scale = 0.0459319413877001f;
    constexpr int key_zp = -18;
    constexpr float softmax_q_scale = 0.00392156862745098f;
    constexpr int softmax_zp = -128;
    constexpr float proj_in_scale = 0.0865f;
    constexpr int proj_in_zp = -9;
    BINEAttentionLowpLayer attn_lowp_layer;

    PermutationVector q_perm{0, 2, 1, 3};
    PermutationVector k_perm{2, 0, 1, 3};
    PermutationVector qkv_o_perm{0, 2, 1, 3};
    attn_lowp_layer.configure(&add_output_tensor,
                              &attn_gamma_weights,
                              &attn_qkv_weights,
                              &attn_qkv_bias,
                              &attn_c_proj_weights,
                              &attn_c_proj_bias,
                              attn_gemm_i_scale,
                              attn_gemm_i_zero,
                              attn_gemm_o_scale,
                              attn_gemm_o_zero,
                              query_scale,
                              query_zp,
                              value_scale,
                              value_zp,
                              key_scale,
                              key_zp,
                              softmax_q_scale,
                              softmax_zp,
                              proj_in_scale,
                              proj_in_zp,
                              q_perm,
                              k_perm,
                              qkv_o_perm,
                              768,
                              16,
                              20,
                              &attn_output_tensor);
    attn_lowp_layer.run();
    // MemAllocTest::print_tensor(attn_output_tensor, "attn_output_tensor");
    BITensor sub_mlp_input;
    sub_mlp_input.allocator()->init(*original_attn_rms_output_tensor.allocator(), gather_output_info);
    BINEArithmeticAddition attn_rms_add; // 注意力RMS相加
    attn_rms_add.configure(&add_output_tensor, &attn_output_tensor, &sub_mlp_input, BIConvertPolicy::SATURATE);
    attn_rms_add.run();
    // MemAllocTest::print_tensor(sub_mlp_input, "mlp_input");
    BINEMLPLayer _mlp_layer; // MLP层
    // 2. 初始化gamma张量
    const std::string &gamma_path = "./input_res/mlp_rms_gamma.npy";
    BITensor gamma = MemAllocTest::create_norm_input(std::vector<int>{768}, gamma_path);
    // 3. 初始化fc_weights的权重
    const std::string &c_fc_weights_path =
            "./input_res/reordered_c_fc_weights.npy";
    std::vector<float> c_fc_weights_scales;
    // 量化信息
    BIQuantizationInfo c_fc_weight_qinfo;
    std::ifstream c_fc_weights_scale_file(
        "./input_res/c_fc_scales.txt");
    float s_value;
    while (c_fc_weights_scale_file >> s_value) {
        c_fc_weights_scales.push_back(s_value);
    }
    BITensor c_fc_weights = MemAllocTest::create_per_channel(c_fc_weights_scales, std::vector{768, 3072},
                                                             c_fc_weight_qinfo, c_fc_weights_path);
    // 4. 初始化fc_bias
    const std::string &c_fc_bias_path = "./input_res/c_fc_bias.npy";
    BITensor c_fc_bias = MemAllocTest::create_norm_bias(3072, c_fc_bias_path);
    // QATTest::print_tensor(c_fc_bias, "c_fc_bias");
    // 5. 输出张量
    BITensor output;
    output.allocator()->init(BITensorInfo(BITensorShape(768, 16, 20), 1, BIDataType::F16));
    output.allocator()->allocate();
    BITensor sub_mlp_output;
    BITensorInfo sub_mlp_output_info = BITensorInfo(BITensorShape(768, seq_len, batch_size), 1, BIDataType::F16);
    sub_mlp_output_info.set_format(Format::F16);
    sub_mlp_output.allocator()->init(*output.allocator(), sub_mlp_output_info);
    // 6. proj的权重
    const std::string &c_proj_path = "./input_res/c_proj_weights.npy";
    BITensor c_proj_weight = MemAllocTest::create_norm_input(std::vector<int>{3072, 768}, c_proj_path);
    // QATTest::print_tensor(c_proj_weight, "c_proj");
    const std::string &c_proj_bias_path =
            "./input_res/c_proj_bias.npy";
    BITensor c_proj_bias = MemAllocTest::create_norm_input(std::vector<int>{768}, c_proj_bias_path);
    // QATTest::print_tensor(c_proj_bias, "c_proj_bias");
    float fc1_input_scale = 0.006902442025203331f;
    int fc1_input_zero_point = -9;
    float fc1_output_scale = 0.1969725440530216f;
    int fc1_output_zero_point = -19;
    float gelu_output_scale = 0.11368115240452337f;
    int gelu_output_zero_point = -127;
    _mlp_layer.configure(&sub_mlp_input, fc1_input_scale,
                         fc1_input_zero_point,
                         &c_fc_weights,
                         &c_fc_bias,
                         &c_fc_weight_qinfo,
                         fc1_output_scale,
                         fc1_output_zero_point,
                         gelu_output_scale,
                         gelu_output_zero_point,
                         &c_proj_weight,
                         &c_proj_bias,
                         &gamma,
                         &sub_mlp_output,
                         20,
                         16);
    _mlp_layer.run(); // 1. 先用输出结果进行相加
    BITensorShape add_output_shape(768, 16, 20);
    BITensor add_output;
    add_output.allocator()->init(BITensorInfo(add_output_shape, 1, BIDataType::F16));
    add_output.allocator()->allocate();
    BITensor sub_add_output;
    sub_add_output.allocator()->init(*add_output.allocator(), gather_output_info);
    BINEArithmeticAddition add_f;
    add_f.configure(&sub_mlp_output, &sub_mlp_input, &sub_add_output, BIConvertPolicy::SATURATE);
    add_f.run();
    // 2. 对结果再进行一次归一化
    BITensor mlp_after_gamma = MemAllocTest::create_norm_input(std::vector{768},
                                                               "./input_res/mlp_after_rms_gamma.npy");
    BITensor mlp_rms_output;
    mlp_rms_output.allocator()->init(BITensorInfo(add_output_shape, 1, BIDataType::F16));
    mlp_rms_output.allocator()->allocate();
    BITensor sub_mlp_rms_output;
    sub_mlp_rms_output.allocator()->init(*mlp_rms_output.allocator(), gather_output_info);
    BINERMSNormLayer rms_norm_layer;
    rms_norm_layer.configure(&sub_add_output, &mlp_after_gamma, &sub_mlp_rms_output);
    rms_norm_layer.run();
    // 3. 对输出结果进行LMHead操作
    BITensor lm_head_weights = MemAllocTest::create_norm_input(std::vector{768, 6003},
                                                               "./input_res/lm_head_weights.npy");
    BITensor lm_head_output;
    lm_head_output.allocator()->init(BITensorInfo(BITensorShape(6003, 16, 20), 1, BIDataType::F16));
    lm_head_output.allocator()->allocate();
    BITensor sub_lm_head_output;
    BITensorInfo sub_lm_head_output_info = BITensorInfo(BITensorShape(6003, seq_len, batch_size), 1, BIDataType::F16);
    sub_lm_head_output_info.set_format(Format::F16);
    sub_lm_head_output.allocator()->init(*lm_head_output.allocator(), sub_lm_head_output_info);

    GEMMInfo gemm_info = GEMMInfo(false,
                                  false,
                                  true,
                                  false,
                                  false,
                                  false,
                                  BIGEMMLowpOutputStageInfo(),
                                  false, true, false,
                                  BIActivationLayerInfo(), false, BIWeightFormat::UNSPECIFIED, false);
    BINEGEMM lm_head_layer;
    lm_head_layer.configure(&sub_mlp_rms_output, &lm_head_weights, nullptr, &sub_lm_head_output, 1.0f, 1.0f, gemm_info);
    lm_head_layer.run();
    MemAllocTest::print_tensor(sub_lm_head_output, "sub_lm_head_output");
    BITensor ids;
    ids.allocator()->init(BITensorInfo(BITensorShape(16, 20), 1, BIDataType::S32));
    ids.allocator()->allocate();
    BITensor sub_ids;
    BITensorInfo sub_ids_info = BITensorInfo(BITensorShape(seq_len, batch_size), 1, BIDataType::S32);
    sub_ids_info.set_format(Format::S32);
    sub_ids.allocator()->init(*ids.allocator(), sub_ids_info);

    BINEArgMinMaxLayer arg_minmax_layer;
    arg_minmax_layer.configure(&sub_lm_head_output, 0, &sub_ids, BIReductionOperation::ARG_IDX_MAX);
    arg_minmax_layer.run();
    // 再次进行运行(动态)
    batch_size = 3;
    seq_len = 4;
    input_tensor_shape = BITensorShape(seq_len, batch_size);
    input_info.set_tensor_shape(input_tensor_shape);
    input_tensor.allocator()->init(*original_input_tensor.allocator(), input_info);
    indices_data = {0, 8, 9, 10, 0, 8, 9, 10, 0, 8, 9, 10};
    MemAllocTest::fill_tensor_val_with_arr(input_tensor, indices_data);
    gather_output_tensor_shape = BITensorShape(768, seq_len, batch_size);
    gather_output_info.set_tensor_shape(gather_output_tensor_shape);
    gather_output_tensor.allocator()->init(*original_gather_output_tensor.allocator(), gather_output_info);
    add_output_tensor.allocator()->init(*original_add_output_tensor.allocator(), gather_output_info);
    sub_add_weight_shape = BITensorShape(768, seq_len);
    sub_add_weight_info.set_tensor_shape(sub_add_weight_shape);
    sub_add_weight.allocator()->init(*add_wte_weight.allocator(), sub_add_weight_info);
    attn_output_tensor.allocator()->init(*attn_origin_o_tensor.allocator(), gather_output_info);
    sub_mlp_input.allocator()->init(*original_attn_rms_output_tensor.allocator(), gather_output_info);
    sub_mlp_output.allocator()->init(*output.allocator(), gather_output_info);
    sub_mlp_rms_output.allocator()->init(*mlp_rms_output.allocator(), gather_output_info);
    sub_add_output.allocator()->init(*add_output.allocator(), gather_output_info);
    sub_lm_head_output_info.set_tensor_shape(BITensorShape(6003, seq_len, batch_size));
    sub_lm_head_output.allocator()->init(*lm_head_output.allocator(), sub_lm_head_output_info);
    sub_ids_info.set_tensor_shape(BITensorShape(seq_len, batch_size));
    sub_ids.allocator()->init(*ids.allocator(), sub_ids_info);

    gather_layer.dynamic_configure(&input_tensor, &gather_output_tensor);
    add_layer.dynamic_configure(&gather_output_tensor, &sub_add_weight, true);
    attn_lowp_layer.dynamic_configure(&add_output_tensor, seq_len, batch_size);
    attn_rms_add.dynamic_configure(&add_output_tensor, &attn_output_tensor, true);
    _mlp_layer.dynamic_configure(&sub_mlp_input, seq_len, batch_size);
    add_f.dynamic_configure(&sub_mlp_output, &sub_mlp_input, false);
    rms_norm_layer.dynamic_configure(&sub_add_output);
    lm_head_layer.dynamic_configure();
    arg_minmax_layer.configure(&sub_lm_head_output, 0, &sub_ids, BIReductionOperation::ARG_IDX_MAX);

    // MemAllocTest::print_tensor(sub_add_weight, "sub_add_weight");
    gather_layer.run();
    add_layer.run();
    attn_lowp_layer.run();
    attn_rms_add.run();
    _mlp_layer.run();
    add_f.run();
    rms_norm_layer.run();
    lm_head_layer.run();
    arg_minmax_layer.run();
    // MemAllocTest::print_tensor(sub_lm_head_output, "sub_lm_head_output");
    MemAllocTest::print_tensor(sub_ids, "ids");
    const auto warmup = 10; // 预热次数
    const auto iterations = 1000; // 运行次数
    const double outlier_threshold = 3.0; // 异常值阈值(标准差倍数)
    std::vector<double> timings;
    timings.reserve(iterations);
    // 预测阶段（不记录时间）
    for (size_t i = 0; i < warmup; ++i) {
        gather_layer.dynamic_configure(&input_tensor, &gather_output_tensor);
        add_layer.dynamic_configure(&gather_output_tensor, &sub_add_weight, true);
        attn_lowp_layer.dynamic_configure(&add_output_tensor, seq_len, batch_size);
        attn_rms_add.dynamic_configure(&add_output_tensor, &attn_output_tensor, true);
        _mlp_layer.dynamic_configure(&sub_mlp_input, seq_len, batch_size);
        add_f.dynamic_configure(&sub_mlp_output, &sub_mlp_input, false);
        rms_norm_layer.dynamic_configure(&sub_add_output);
        lm_head_layer.dynamic_configure();
        gather_layer.dynamic_configure(&input_tensor, &gather_output_tensor);
        add_layer.dynamic_configure(&gather_output_tensor, &sub_add_weight, true);
        attn_lowp_layer.dynamic_configure(&add_output_tensor, seq_len, batch_size);
        attn_rms_add.dynamic_configure(&add_output_tensor, &attn_output_tensor, true);
        _mlp_layer.dynamic_configure(&sub_mlp_input, seq_len, batch_size);
        add_f.dynamic_configure(&sub_mlp_output, &sub_mlp_input, false);
        rms_norm_layer.dynamic_configure(&sub_add_output);
        gather_layer.run();
        add_layer.run();
        attn_lowp_layer.run();
        attn_rms_add.run();
        _mlp_layer.run();
        add_f.run();
        rms_norm_layer.run();
        lm_head_layer.run();
    } // 正式测量
    for (size_t i = 0; i < iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        gather_layer.dynamic_configure(&input_tensor, &gather_output_tensor);
        add_layer.dynamic_configure(&gather_output_tensor, &sub_add_weight, true);
        attn_lowp_layer.dynamic_configure(&add_output_tensor, seq_len, batch_size);
        attn_rms_add.dynamic_configure(&add_output_tensor, &attn_output_tensor, true);
        _mlp_layer.dynamic_configure(&sub_mlp_input, seq_len, batch_size);
        add_f.dynamic_configure(&sub_mlp_output, &sub_mlp_input, false);
        rms_norm_layer.dynamic_configure(&sub_add_output);
        lm_head_layer.dynamic_configure();
        gather_layer.dynamic_configure(&input_tensor, &gather_output_tensor);
        add_layer.dynamic_configure(&gather_output_tensor, &sub_add_weight, true);
        attn_lowp_layer.dynamic_configure(&add_output_tensor, seq_len, batch_size);
        attn_rms_add.dynamic_configure(&add_output_tensor, &attn_output_tensor, true);
        _mlp_layer.dynamic_configure(&sub_mlp_input, seq_len, batch_size);
        add_f.dynamic_configure(&sub_mlp_output, &sub_mlp_input, false);
        rms_norm_layer.dynamic_configure(&sub_add_output);
        gather_layer.run();
        add_layer.run();
        attn_lowp_layer.run();
        attn_rms_add.run();
        _mlp_layer.run();
        add_f.run();
        rms_norm_layer.run();
        lm_head_layer.run();
        auto end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double, std::milli>(end - start).count();
        timings.push_back(duration);
    }
    // 异常值过滤
    auto result = [&] {
        double sum = std::accumulate(timings.begin(), timings.end(), 0.0);
        double mean = sum / timings.size();
        double sq_sum = std::inner_product(timings.begin(), timings.end(),
                                           timings.begin(), 0.0);
        double stdev = std::sqrt(sq_sum / timings.size() - mean * mean);
        return std::make_pair(mean, stdev);
    }();
    double avg = result.first;
    double std_dev = result.second;
    // 应用3-sigma法则过滤异常值
    std::vector<double> filtered;
    std::copy_if(timings.begin(), timings.end(), std::back_inserter(filtered),
                 [=](double x) { return std::abs(x - avg) < outlier_threshold * std_dev; });
    // 重新计算统计量
    double valid_avg = std::accumulate(filtered.begin(), filtered.end(), 0.0) / filtered.size();
    auto [min_it, max_it] = std::minmax_element(filtered.begin(), filtered.end());
    auto perf_status = MemAllocTest::PerfStats{
        valid_avg,
        std_dev,
        *min_it,
        *max_it,
        filtered.size()
    };

    std::cout << "Performance Report:\n"
            << "Iterations: " << perf_status.iterations << "\n"
            << "Avg Time:   " << perf_status.avg_ms << " ms\n"
            << "Std Dev:    " << perf_status.std_dev_ms << " ms\n"
            << "Min Time:   " << perf_status.min_ms << " ms\n"
            << "Max Time:   " << perf_status.max_ms << " ms\n";
}

TEST(MemAllocGPT2Origin, GPT2AlloctOrigin) {
    using namespace BatmanInfer;
    // BIScheduler::set(BIScheduler::Type::OMP);
    BIScheduler::get().set_num_threads(std::thread::hardware_concurrency());
    BIMemoryGroup group{BIMemoryManagerOnDemand::make_default()};

    int batch_size = 1;
    int seq_len = 1;
    // 1. 初始化一个最大input算子
    BITensor original_input_tensor;
    BITensorShape original_input_tensor_shape(16, 20);
    original_input_tensor.allocator()->init(BITensorInfo(original_input_tensor_shape, 1, BIDataType::U32));
    original_input_tensor.allocator()->allocate();

    // 1.1 初始化一个小型算子
    BITensor input_tensor;
    BITensorShape input_tensor_shape(seq_len, batch_size);
    BITensorInfo input_info(input_tensor_shape, 1, BIDataType::U32);
    input_info.set_format(Format::U32);
    input_tensor.allocator()->init(*original_input_tensor.allocator(), input_info);
    std::vector<uint32_t> indices_data{0, 1, 2};
    MemAllocTest::fill_tensor_val_with_arr(input_tensor, indices_data);


    // 2. Gather的权重
    BITensorShape gather_weight_shape(768, 6003);
    const std::string &weight_path =
            "./input_res/transformer_wte_weight.npy";
    BITensor weight = utils::create_type_tensor(
        weight_path, gather_weight_shape,
        BIDataType::F16);

    // 3. 输出原始矩阵
    BITensor original_gather_output_tensor;
    BITensorShape original_gather_output_tensor_shape(768, 16, 20);
    original_gather_output_tensor.allocator()->init(
        BITensorInfo(original_gather_output_tensor_shape, 1, BIDataType::F16));
    original_gather_output_tensor.allocator()->allocate();
    BITensor original_attn_rms_output_tensor;
    original_attn_rms_output_tensor.allocator()->init(
        BITensorInfo(original_gather_output_tensor_shape, 1, BIDataType::F16));
    original_attn_rms_output_tensor.allocator()->allocate();
    // 3.1 输出矩阵的子矩阵
    BITensor gather_output_tensor;
    BITensorShape gather_output_tensor_shape(768, seq_len, batch_size);
    BITensorInfo gather_output_info(gather_output_tensor_shape, 1, BIDataType::F16);
    gather_output_info.set_format(Format::F16);
    gather_output_tensor.allocator()->init(*original_gather_output_tensor.allocator(), gather_output_info);
    // 2. 进行NEGather筛选
    BINEGather gather_layer;
    gather_layer.configure(&weight, &input_tensor, &gather_output_tensor, 1);
    gather_layer.run();
    // 3. Add权重的获取
    BITensorShape add_wte_weight_shape(768, 16);
    const std::string &add_wte_weight_path = "./input_res/add_wte_weights.npy";
    BITensor add_wte_weight = utils::create_type_tensor(
        add_wte_weight_path, add_wte_weight_shape,
        BIDataType::F16);

    // 临时的数据
    BITensor sub_add_weight;
    BITensorShape sub_add_weight_shape(768, seq_len);
    BITensorInfo sub_add_weight_info(sub_add_weight_shape, 1, BIDataType::F16);
    sub_add_weight_info.set_format(Format::F16);
    sub_add_weight.allocator()->init(*add_wte_weight.allocator(), sub_add_weight_info);
    // 4. Add输出的原始最大值
    BITensor original_add_output_tensor;
    original_add_output_tensor.allocator()->init(
        BITensorInfo(original_gather_output_tensor_shape, 1, BIDataType::F16));
    original_add_output_tensor.allocator()->allocate();
    // 4.1 Add输出的新的数据格式
    BITensor add_output_tensor;
    add_output_tensor.allocator()->init(*original_add_output_tensor.allocator(), gather_output_info);
    BINEArithmeticAddition add_layer;
    add_layer.configure(&gather_output_tensor, &sub_add_weight, &add_output_tensor, BIConvertPolicy::SATURATE);
    add_layer.run();
    // 5 获取Attention模块的权重
    // 5.1 gamma权重
    BITensorShape attn_gamma_weights_shape(768);
    const std::string &gamma_weights_path = "./input_res/attn_gamma_weights.npy";
    BITensor attn_gamma_weights = utils::create_type_tensor(
        gamma_weights_path, attn_gamma_weights_shape,
        BIDataType::F16);
    BITensor attn_origin_o_tensor;
    attn_origin_o_tensor.allocator()->init(BITensorInfo(original_gather_output_tensor_shape, 1, BIDataType::F16));
    attn_origin_o_tensor.allocator()->allocate();
    BITensor attn_output_tensor;
    attn_output_tensor.allocator()->init(*attn_origin_o_tensor.allocator(), gather_output_info);
    // 5.2 c_attn权重和偏置值
    BITensorShape attn_qkv_weights_shape(2304, 768);
    const std::string &attn_qkv_weights_path = "./input_res/c_attn_weights.npy";
    BITensor attn_qkv_weights = utils::create_type_tensor(attn_qkv_weights_path, attn_qkv_weights_shape,
                                                          BIDataType::F16);

    BITensorShape attn_qkv_bias_shape(2304);
    const std::string &attn_qkv_bias_path = "./input_res/c_attn_bias.npy";
    BITensor attn_qkv_bias = utils::create_type_tensor(attn_qkv_bias_path, attn_qkv_bias_shape,
                                                       BIDataType::F16);
    BITensorShape attn_c_proj_weights_shape(768, 768);
    const std::string &attn_c_proj_weights_path = "./input_res/p_attn_weights.npy";
    BITensor attn_c_proj_weights = utils::create_type_tensor(attn_c_proj_weights_path, attn_c_proj_weights_shape,
                                                             BIDataType::F16);
    BITensorShape attn_c_proj_bias_shape(768);
    const std::string &attn_c_proj_bias_path = "./input_res/p_attn_bias.npy";
    BITensor attn_c_proj_bias = utils::create_type_tensor(attn_c_proj_bias_path, attn_c_proj_bias_shape,
                                                          BIDataType::F16);
    BINEAttentionLayer attn_layer;

    PermutationVector q_perm{0, 2, 1, 3};
    PermutationVector k_perm{2, 0, 1, 3};
    PermutationVector qkv_o_perm{0, 2, 1, 3};
    attn_layer.configure(&add_output_tensor,
                         &attn_gamma_weights,
                         &attn_qkv_weights,
                         &attn_qkv_bias,
                         &attn_c_proj_weights,
                         &attn_c_proj_bias, q_perm,
                         k_perm,
                         qkv_o_perm,
                         768,
                         16,
                         20,
                         &attn_output_tensor);
    attn_layer.run();
    BITensor sub_mlp_input;
    sub_mlp_input.allocator()->init(*original_attn_rms_output_tensor.allocator(), gather_output_info);
    BINEArithmeticAddition attn_rms_add; // 注意力RMS相加
    attn_rms_add.configure(&add_output_tensor, &attn_output_tensor, &sub_mlp_input, BIConvertPolicy::SATURATE);
    attn_rms_add.run();


    BINEFeedForwardLayer _mlp_layer; // MLP层
    //2. 初始化gamma张量
    const std::string &gamma_path = "./input_res/mlp_rms_gamma.npy";
    BITensor gamma = MemAllocTest::create_norm_input(std::vector<int>{768}, gamma_path);
    // 3. 初始化fc_weights的权重
    const std::string &c_fc_weights_path =
            "./input_res/reordered_c_fc_weights.npy";
    BITensor c_fc_weights = utils::create_type_tensor(c_fc_weights_path, BITensorShape(3072, 768), BIDataType::F16);
    // 4. 初始化fc_bias
    const std::string &c_fc_bias_path = "./input_res/c_fc_bias.npy";
    BITensor c_fc_bias = utils::create_type_tensor(c_fc_bias_path, BITensorShape(3072), BIDataType::F16);
    // 5. 输出张量
    BITensor output;
    output.allocator()->init(BITensorInfo(BITensorShape(768, 16, 20), 1, BIDataType::F16));
    output.allocator()->allocate();
    BITensor sub_mlp_output;
    BITensorInfo sub_mlp_output_info = BITensorInfo(BITensorShape(768, seq_len, batch_size), 1, BIDataType::F16);
    sub_mlp_output_info.set_format(Format::F16);
    sub_mlp_output.allocator()->init(*output.allocator(), sub_mlp_output_info);
    // 6. proj的权重
    const std::string &c_proj_path = "./input_res/c_proj_weights.npy";
    BITensor c_proj_weight = MemAllocTest::create_norm_input(std::vector<int>{3072, 768}, c_proj_path);
    const std::string &c_proj_bias_path =
            "./input_res/c_proj_bias.npy";
    BITensor c_proj_bias = MemAllocTest::create_norm_input(std::vector<int>{768}, c_proj_bias_path);
    const BIActivationLayerInfo act_info(BIActivationFunction::GELU);
    _mlp_layer.configure(&sub_mlp_input,
                         &c_fc_weights,
                         &c_fc_bias,
                         &c_proj_weight,
                         &c_proj_bias,
                         &gamma,
                         act_info,
                         &sub_mlp_output,
                         20,
                         16);
    _mlp_layer.run(); // 1. 先用输出结果进行相加
    BITensorShape add_output_shape(768, 16, 20);
    BITensor add_output;
    add_output.allocator()->init(BITensorInfo(add_output_shape, 1, BIDataType::F16));
    add_output.allocator()->allocate();
    BITensor sub_add_output;
    sub_add_output.allocator()->init(*add_output.allocator(), gather_output_info);
    BINEArithmeticAddition add_f;
    add_f.configure(&sub_mlp_output, &sub_mlp_input, &sub_add_output, BIConvertPolicy::SATURATE);
    add_f.run();
    // 2. 对结果再进行一次归一化
    BITensor mlp_after_gamma = MemAllocTest::create_norm_input(std::vector{768},
                                                               "./input_res/mlp_after_rms_gamma.npy");
    BITensor mlp_rms_output;
    mlp_rms_output.allocator()->init(BITensorInfo(add_output_shape, 1, BIDataType::F16));
    mlp_rms_output.allocator()->allocate();
    BITensor sub_mlp_rms_output;
    sub_mlp_rms_output.allocator()->init(*mlp_rms_output.allocator(), gather_output_info);
    BINERMSNormLayer rms_norm_layer;
    rms_norm_layer.configure(&sub_add_output, &mlp_after_gamma, &sub_mlp_rms_output);
    rms_norm_layer.run();
    // 3. 对输出结果进行LMHead操作
    BITensor lm_head_weights = MemAllocTest::create_norm_input(std::vector{768, 6003},
                                                               "./input_res/lm_head_weights.npy");
    BITensor lm_head_output;
    lm_head_output.allocator()->init(BITensorInfo(BITensorShape(6003, 16, 20), 1, BIDataType::F16));
    lm_head_output.allocator()->allocate();
    BITensor sub_lm_head_output;
    BITensorInfo sub_lm_head_output_info = BITensorInfo(BITensorShape(6003, seq_len, batch_size), 1, BIDataType::F16);
    sub_lm_head_output_info.set_format(Format::F16);
    sub_lm_head_output.allocator()->init(*lm_head_output.allocator(), sub_lm_head_output_info);

    GEMMInfo gemm_info = GEMMInfo(false,
                                  false,
                                  true,
                                  false,
                                  false,
                                  false,
                                  BIGEMMLowpOutputStageInfo(),
                                  false, true, false,
                                  BIActivationLayerInfo(), false, BIWeightFormat::UNSPECIFIED, false);
    BINEGEMM lm_head_layer;
    lm_head_layer.configure(&sub_mlp_rms_output, &lm_head_weights, nullptr, &sub_lm_head_output, 1.0f, 1.0f, gemm_info);
    lm_head_layer.run();
    // MemAllocTest::print_tensor(sub_mlp_output, "attn_output_tensor");
    BITensor ids;
    ids.allocator()->init(BITensorInfo(BITensorShape(16, 20), 1, BIDataType::S32));
    ids.allocator()->allocate();
    BITensor sub_ids;
    BITensorInfo sub_ids_info = BITensorInfo(BITensorShape(seq_len, batch_size), 1, BIDataType::S32);
    sub_ids_info.set_format(Format::S32);
    sub_ids.allocator()->init(*ids.allocator(), sub_ids_info);

    BINEArgMinMaxLayer arg_minmax_layer;
    arg_minmax_layer.configure(&sub_lm_head_output, 0, &sub_ids, BIReductionOperation::ARG_IDX_MAX);
    arg_minmax_layer.run();
    MemAllocTest::print_tensor(sub_ids, "ids");


    // 再次进行运行(动态)
    batch_size = 1;
    seq_len = 3;
    input_tensor_shape = BITensorShape(seq_len, batch_size);
    input_info.set_tensor_shape(input_tensor_shape);
    input_tensor.allocator()->init(*original_input_tensor.allocator(), input_info);
    indices_data = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    MemAllocTest::fill_tensor_val_with_arr(input_tensor, indices_data);
    gather_output_tensor_shape = BITensorShape(768, seq_len, batch_size);
    gather_output_info.set_tensor_shape(gather_output_tensor_shape);
    gather_output_tensor.allocator()->init(*original_gather_output_tensor.allocator(), gather_output_info);
    add_output_tensor.allocator()->init(*original_add_output_tensor.allocator(), gather_output_info);
    sub_add_weight_shape = BITensorShape(768, seq_len);
    sub_add_weight_info.set_tensor_shape(sub_add_weight_shape);
    sub_add_weight.allocator()->init(*add_wte_weight.allocator(), sub_add_weight_info);
    attn_output_tensor.allocator()->init(*attn_origin_o_tensor.allocator(), gather_output_info);
    sub_mlp_input.allocator()->init(*original_attn_rms_output_tensor.allocator(), gather_output_info);
    sub_mlp_output.allocator()->init(*output.allocator(), gather_output_info);
    sub_mlp_rms_output.allocator()->init(*mlp_rms_output.allocator(), gather_output_info);
    sub_add_output.allocator()->init(*add_output.allocator(), gather_output_info);
    sub_lm_head_output_info.set_tensor_shape(BITensorShape(6003, seq_len, batch_size));
    sub_lm_head_output.allocator()->init(*lm_head_output.allocator(), sub_lm_head_output_info);
    sub_ids_info.set_tensor_shape(BITensorShape(seq_len, batch_size));
    sub_ids.allocator()->init(*ids.allocator(), sub_ids_info);
    gather_layer.dynamic_configure(&input_tensor, &gather_output_tensor);
    add_layer.dynamic_configure(&gather_output_tensor, &sub_add_weight, true);
    attn_layer.dynamic_configure(&add_output_tensor, seq_len, batch_size);
    attn_rms_add.dynamic_configure(&add_output_tensor, &attn_output_tensor, true);
    _mlp_layer.dynamic_configure(&sub_mlp_input, seq_len, batch_size);
    add_f.dynamic_configure(&sub_mlp_output, &sub_mlp_input, false);
    rms_norm_layer.dynamic_configure(&sub_add_output);
    lm_head_layer.dynamic_configure();
    arg_minmax_layer.configure(&sub_lm_head_output, 0, &sub_ids, BIReductionOperation::ARG_IDX_MAX);
    // // //
    // // MemAllocTest::print_tensor(sub_add_weight, "sub_add_weight");
    gather_layer.run();
    add_layer.run();
    attn_layer.run();
    attn_rms_add.run();
    _mlp_layer.run();
    add_f.run();
    rms_norm_layer.run();
    lm_head_layer.run();
    arg_minmax_layer.run();
    MemAllocTest::print_tensor(sub_ids, "ids");
    MemAllocTest::print_tensor(sub_lm_head_output, "sub_lm_head_output");

    batch_size = 3;
    seq_len = 3;
    input_tensor_shape = BITensorShape(seq_len, batch_size);
    input_info.set_tensor_shape(input_tensor_shape);
    input_tensor.allocator()->init(*original_input_tensor.allocator(), input_info);
    indices_data = {0, 2, 2, 0, 1, 2, 0, 1, 2};
    MemAllocTest::fill_tensor_val_with_arr(input_tensor, indices_data);
    gather_output_tensor_shape = BITensorShape(768, seq_len, batch_size);
    gather_output_info.set_tensor_shape(gather_output_tensor_shape);
    gather_output_tensor.allocator()->init(*original_gather_output_tensor.allocator(), gather_output_info);
    add_output_tensor.allocator()->init(*original_add_output_tensor.allocator(), gather_output_info);
    sub_add_weight_shape = BITensorShape(768, seq_len);
    sub_add_weight_info.set_tensor_shape(sub_add_weight_shape);
    sub_add_weight.allocator()->init(*add_wte_weight.allocator(), sub_add_weight_info);
    attn_output_tensor.allocator()->init(*attn_origin_o_tensor.allocator(), gather_output_info);
    sub_mlp_input.allocator()->init(*original_attn_rms_output_tensor.allocator(), gather_output_info);
    sub_mlp_output.allocator()->init(*output.allocator(), gather_output_info);
    sub_mlp_rms_output.allocator()->init(*mlp_rms_output.allocator(), gather_output_info);
    sub_add_output.allocator()->init(*add_output.allocator(), gather_output_info);
    sub_lm_head_output_info.set_tensor_shape(BITensorShape(6003, seq_len, batch_size));
    sub_lm_head_output.allocator()->init(*lm_head_output.allocator(), sub_lm_head_output_info);
    sub_ids_info.set_tensor_shape(BITensorShape(seq_len, batch_size));
    sub_ids.allocator()->init(*ids.allocator(), sub_ids_info);
    gather_layer.dynamic_configure(&input_tensor, &gather_output_tensor);
    add_layer.dynamic_configure(&gather_output_tensor, &sub_add_weight, true);
    attn_layer.dynamic_configure(&add_output_tensor, seq_len, batch_size);
    attn_rms_add.dynamic_configure(&add_output_tensor, &attn_output_tensor, true);
    _mlp_layer.dynamic_configure(&sub_mlp_input, seq_len, batch_size);
    add_f.dynamic_configure(&sub_mlp_output, &sub_mlp_input, false);
    rms_norm_layer.dynamic_configure(&sub_add_output);
    lm_head_layer.dynamic_configure();
    arg_minmax_layer.configure(&sub_lm_head_output, 0, &sub_ids, BIReductionOperation::ARG_IDX_MAX);
    gather_layer.run();
    add_layer.run();
    attn_layer.run();
    attn_rms_add.run();
    _mlp_layer.run();
    add_f.run();
    rms_norm_layer.run();
    lm_head_layer.run();
    arg_minmax_layer.run();
    MemAllocTest::print_tensor(sub_ids, "ids");
    // MemAllocTest::print_tensor(sub_lm_head_output, "sub_lm_head_output");

    batch_size = 5;
    seq_len = 3;
    input_tensor_shape = BITensorShape(seq_len, batch_size);
    input_info.set_tensor_shape(input_tensor_shape);
    input_tensor.allocator()->init(*original_input_tensor.allocator(), input_info);
    indices_data = {0, 2, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2};
    MemAllocTest::fill_tensor_val_with_arr(input_tensor, indices_data);
    gather_output_tensor_shape = BITensorShape(768, seq_len, batch_size);
    gather_output_info.set_tensor_shape(gather_output_tensor_shape);
    gather_output_tensor.allocator()->init(*original_gather_output_tensor.allocator(), gather_output_info);
    add_output_tensor.allocator()->init(*original_add_output_tensor.allocator(), gather_output_info);
    sub_add_weight_shape = BITensorShape(768, seq_len);
    sub_add_weight_info.set_tensor_shape(sub_add_weight_shape);
    sub_add_weight.allocator()->init(*add_wte_weight.allocator(), sub_add_weight_info);
    attn_output_tensor.allocator()->init(*attn_origin_o_tensor.allocator(), gather_output_info);
    sub_mlp_input.allocator()->init(*original_attn_rms_output_tensor.allocator(), gather_output_info);
    sub_mlp_output.allocator()->init(*output.allocator(), gather_output_info);
    sub_mlp_rms_output.allocator()->init(*mlp_rms_output.allocator(), gather_output_info);
    sub_add_output.allocator()->init(*add_output.allocator(), gather_output_info);
    sub_lm_head_output_info.set_tensor_shape(BITensorShape(6003, seq_len, batch_size));
    sub_lm_head_output.allocator()->init(*lm_head_output.allocator(), sub_lm_head_output_info);
    sub_ids_info.set_tensor_shape(BITensorShape(seq_len, batch_size));
    sub_ids.allocator()->init(*ids.allocator(), sub_ids_info);
    gather_layer.dynamic_configure(&input_tensor, &gather_output_tensor);
    add_layer.dynamic_configure(&gather_output_tensor, &sub_add_weight, true);
    attn_layer.dynamic_configure(&add_output_tensor, seq_len, batch_size);
    attn_rms_add.dynamic_configure(&add_output_tensor, &attn_output_tensor, true);
    _mlp_layer.dynamic_configure(&sub_mlp_input, seq_len, batch_size);
    add_f.dynamic_configure(&sub_mlp_output, &sub_mlp_input, false);
    rms_norm_layer.dynamic_configure(&sub_add_output);
    lm_head_layer.dynamic_configure();
    arg_minmax_layer.configure(&sub_lm_head_output, 0, &sub_ids, BIReductionOperation::ARG_IDX_MAX);


    gather_layer.run();
    add_layer.run();
    attn_layer.run();
    attn_rms_add.run();
    _mlp_layer.run();
    add_f.run();
    rms_norm_layer.run();
    lm_head_layer.run();
    arg_minmax_layer.run();
    MemAllocTest::print_tensor(sub_ids, "ids");
    // MemAllocTest::print_tensor(sub_lm_head_output, "sub_lm_head_output");
    // const auto warmup = 10; // 预热次数
    // const auto iterations = 1000; // 运行次数
    // const double outlier_threshold = 3.0; // 异常值阈值(标准差倍数)
    // std::vector<double> timings;
    // timings.reserve(iterations);
    // // 预测阶段（不记录时间）
    // for (size_t i = 0; i < warmup; ++i) {
    //     gather_layer.dynamic_configure(&input_tensor, &gather_output_tensor);
    //     add_layer.dynamic_configure(&gather_output_tensor, &sub_add_weight, true);
    //     attn_layer.dynamic_configure(&add_output_tensor, seq_len, batch_size);
    //     attn_rms_add.dynamic_configure(&add_output_tensor, &attn_output_tensor, true);
    //     _mlp_layer.dynamic_configure(&sub_mlp_input, seq_len, batch_size);
    //     add_f.dynamic_configure(&sub_mlp_output, &sub_mlp_input, false);
    //     rms_norm_layer.dynamic_configure(&sub_add_output);
    //     lm_head_layer.dynamic_configure();
    //     gather_layer.dynamic_configure(&input_tensor, &gather_output_tensor);
    //     add_layer.dynamic_configure(&gather_output_tensor, &sub_add_weight, true);
    //     attn_layer.dynamic_configure(&add_output_tensor, seq_len, batch_size);
    //     attn_rms_add.dynamic_configure(&add_output_tensor, &attn_output_tensor, true);
    //     _mlp_layer.dynamic_configure(&sub_mlp_input, seq_len, batch_size);
    //     add_f.dynamic_configure(&sub_mlp_output, &sub_mlp_input, false);
    //     rms_norm_layer.dynamic_configure(&sub_add_output);
    //     gather_layer.run();
    //     add_layer.run();
    //     attn_layer.run();
    //     attn_rms_add.run();
    //     _mlp_layer.run();
    //     add_f.run();
    //     rms_norm_layer.run();
    //     lm_head_layer.run();
    // } // 正式测量
    // for (size_t i = 0; i < iterations; ++i) {
    //     auto start = std::chrono::high_resolution_clock::now();
    //     gather_layer.dynamic_configure(&input_tensor, &gather_output_tensor);
    //     add_layer.dynamic_configure(&gather_output_tensor, &sub_add_weight, true);
    //     attn_layer.dynamic_configure(&add_output_tensor, seq_len, batch_size);
    //     attn_rms_add.dynamic_configure(&add_output_tensor, &attn_output_tensor, true);
    //     _mlp_layer.dynamic_configure(&sub_mlp_input, seq_len, batch_size);
    //     add_f.dynamic_configure(&sub_mlp_output, &sub_mlp_input, false);
    //     rms_norm_layer.dynamic_configure(&sub_add_output);
    //     lm_head_layer.dynamic_configure();
    //     gather_layer.dynamic_configure(&input_tensor, &gather_output_tensor);
    //     add_layer.dynamic_configure(&gather_output_tensor, &sub_add_weight, true);
    //     attn_layer.dynamic_configure(&add_output_tensor, seq_len, batch_size);
    //     attn_rms_add.dynamic_configure(&add_output_tensor, &attn_output_tensor, true);
    //     _mlp_layer.dynamic_configure(&sub_mlp_input, seq_len, batch_size);
    //     add_f.dynamic_configure(&sub_mlp_output, &sub_mlp_input, false);
    //     rms_norm_layer.dynamic_configure(&sub_add_output);
    //     gather_layer.run();
    //     add_layer.run();
    //     attn_layer.run();
    //     attn_rms_add.run();
    //     _mlp_layer.run();
    //     add_f.run();
    //     rms_norm_layer.run();
    //     lm_head_layer.run();
    //     auto end = std::chrono::high_resolution_clock::now();
    //     double duration = std::chrono::duration<double, std::milli>(end - start).count();
    //     timings.push_back(duration);
    // }
    // // 异常值过滤
    // auto result = [&] {
    //     double sum = std::accumulate(timings.begin(), timings.end(), 0.0);
    //     double mean = sum / timings.size();
    //     double sq_sum = std::inner_product(timings.begin(), timings.end(),
    //                                        timings.begin(), 0.0);
    //     double stdev = std::sqrt(sq_sum / timings.size() - mean * mean);
    //     return std::make_pair(mean, stdev);
    // }();
    // double avg = result.first;
    // double std_dev = result.second;
    // // 应用3-sigma法则过滤异常值
    // std::vector<double> filtered;
    // std::copy_if(timings.begin(), timings.end(), std::back_inserter(filtered),
    //              [=](double x) { return std::abs(x - avg) < outlier_threshold * std_dev; });
    // // 重新计算统计量
    // double valid_avg = std::accumulate(filtered.begin(), filtered.end(), 0.0) / filtered.size();
    // auto [min_it, max_it] = std::minmax_element(filtered.begin(), filtered.end());
    // auto perf_status = MemAllocTest::PerfStats{
    //     valid_avg,
    //     std_dev,
    //     *min_it,
    //     *max_it,
    //     filtered.size()
    // };
    //
    // std::cout << "Performance Report:\n"
    //         << "Iterations: " << perf_status.iterations << "\n"
    //         << "Avg Time:   " << perf_status.avg_ms << " ms\n"
    //         << "Std Dev:    " << perf_status.std_dev_ms << " ms\n"
    //         << "Min Time:   " << perf_status.min_ms << " ms\n"
    //         << "Max Time:   " << perf_status.max_ms << " ms\n";
}
