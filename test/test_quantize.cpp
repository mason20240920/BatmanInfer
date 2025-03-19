//
// Created by Mason on 2025/3/11.
//
#include <runtime/neon/bi_ne_functions.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "data/core/utils/quantization/asymm_helpers.hpp"

using namespace BatmanInfer;

namespace QATTest {
    void invert_qinfo_offset(BITensor &t) {
        BIQuantizationInfo qinfo = t.info()->quantization_info();
        t.info()->set_quantization_info(BIQuantizationInfo(qinfo.scale()[0], -qinfo.offset()[0], qinfo.is_dynamic()));
    }

    void print_tensor(const BatmanInfer::BITensor &tensor, const std::string &name = "temp") {
        std::cout << name << std::endl;
        BatmanInfer::BIIOFormatInfo format;
        format.element_delim = ", "; // 元素之间用逗号分隔
        format.row_delim = "\n"; // 每行换行
        format.align_columns = 1; // 对齐列

        tensor.print(std::cout, format);
    }

    // 将浮点scale转换为定点数乘数和移位值
    void calculate_multiplier_and_shift(float scale, int32_t &multiplier, int32_t &shift) {
        // 1. 找到合适的移位值，使scale * 2^31接近但不超过INT32_MAX
        shift = 0;
        float scale_abs = std::abs(scale);

        while (scale_abs * (1LL << 31) > static_cast<float>(INT32_MAX)) {
            shift++;
            scale_abs /= 2.0f;
        }

        while (scale_abs * (1LL << 31) < static_cast<float>(INT32_MAX) / 2) {
            shift--;
            scale_abs *= 2.0f;
        }

        // 2. 计算定点数乘数
        int32_t q31_mul = static_cast<int32_t>(std::round(scale_abs * (1LL << 31)));
        multiplier = (scale >= 0) ? q31_mul : -q31_mul;

        // 确保shift为正数（右移）
        if (shift < 0) {
            multiplier <<= -shift;
            shift = 0;
        }
    }

    void configure_gemm_output_stage(BIITensor *input, BIITensor *output,
                                     float input_scale,
                                     const std::vector<float> &weight_scales,
                                     float output_scale,
                                     int32_t output_offset) {
        // 1. 创建output stage配置
        BIGEMMLowpOutputStageInfo output_stage;
        output_stage.type = BIGEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT;
        output_stage.output_data_type = BIDataType::QASYMM8_SIGNED;

        // 2. 计算每个通道的combined scale
        const int N = input->info()->dimension(1);
        std::vector<int32_t> multipliers(N);
        std::vector<int32_t> shifts(N);

        for (int i = 0; i < N; ++i) {
            // 计算combined scale: (input_scale * weight_scale) / output_scale
            float combined_scale = (input_scale * weight_scales[i]) / output_scale;

            // 将浮点scale转换为定点数参数
            int32_t multiplier;
            int32_t shift;
            quantization::calculate_quantized_multiplier(
                combined_scale, &multiplier, &shift);

            multipliers[i] = multiplier;
            shifts[i] = shift;
        }

        // 3. 设置output stage参数
        output_stage.gemmlowp_multipliers = multipliers;
        output_stage.gemmlowp_shifts = shifts;
        output_stage.gemmlowp_offset = output_offset;
        // output_stage.is_per_channel = true; // 使用per-channel量化

        // 4. 创建并配置NEGEMMLowpOutputStage
        BINEGEMMLowpOutputStage output_stage_kernel;
        output_stage_kernel.configure(input, nullptr, output, output_stage);
    }

    // 假设我们有以下输入：
    // - input: ITensor 包含S32结果 [M, N]
    // - input_scale: float 输入的缩放因子
    // - weight_scales: 包含N个元素的vector<float>，每个通道的权重缩放因子

    void dequantize_gemm_output(const BIITensor *input, float input_scale,
                                const std::vector<float> &weight_scales,
                                BIITensor *output) {
        // 1. 预计算每个通道的combined_scale
        std::vector<float> combined_scales(weight_scales.size());
        for (size_t i = 0; i < weight_scales.size(); ++i) {
            combined_scales[i] = input_scale * weight_scales[i];
        }

        // // 2. 将combined_scales转换为量化乘数和移位值
        // std::vector<int32_t> multipliers(weight_scales.size());
        // std::vector<int32_t> shifts(weight_scales.size());
        // for (size_t i = 0; i < weight_scales.size(); ++i) {
        //     calculate_multiplier_and_shift(combined_scales[i], multipliers[i], shifts[i]);
        // }

        // 3. 对每个输出通道应用反量化
        const auto *input_ptr = reinterpret_cast<const int32_t *>(input->buffer());
        auto *output_ptr = reinterpret_cast<float *>(output->buffer());

        const int M = input->info()->dimension(0);
        const int N = input->info()->dimension(1);

        for (int n = 0; n < N; ++n) {
            // const int32_t multiplier = multipliers[n];
            // const int32_t shift = shifts[n];

            for (int m = 0; m < M; ++m) {
                const int32_t value = input_ptr[m * N + n];
                // // 使用定点乘法和移位进行反量化
                // const int32_t scaled = quantization::multiply_by_quantized_multiplier(value, multiplier, shift);
                // 转换为浮点数
                output_ptr[m * N + n] = static_cast<float>(value) * combined_scales[n];
            }
        }
    }


    /**
     * @brief gemmlowp 矩阵计算
     * @param input_data  QASYMM8输入数据
     * @param weights_data QSYMM8_PER_CHANNEL权重数据
     * @param input_scale 输入scale
     * @param input_offset 输入zero point
     * @param weight_scales_tmp 权重每通道scale
     * @param M 矩阵维度
     * @param K
     * @param N
     * @param num_channels 通道数
     */
    void gemmlowp_mixed_quantized(
        const int8_t *input_data,
        const int8_t *weights_data,
        const int32_t *bias_data,
        float input_scale,
        int input_offset,
        const std::vector<float> &weight_scales,
        float output_scale,
        int output_offset,
        int M, int K, int N,
        int num_channels) {
        // 1. 创建输入tensor (QASYMM8)
        BIQuantizationInfo input_qinfo = BIQuantizationInfo(input_scale, input_offset);
        BITensorShape input_shape(K, M); // [M, K]
        BITensor input;
        auto input_info = BITensorInfo(input_shape, 1, BIDataType::QASYMM8_SIGNED, input_qinfo);
        input.allocator()->init(input_info);

        // 2. 创建权重tensor (QSYMM8_PER_CHANNEL)
        // 为每个通道创建scale vector
        // const std::vector<float> &weight_scales_vec = weight_scales_tmp;
        // BIQuantizationInfo weight_info = BIQuantizationInfo(weight_scales_vec);
        BIQuantizationInfo weights_qinfo = BIQuantizationInfo(weight_scales);
        BITensorShape weights_shape(N, K); // [K, N]
        BITensor weights;
        auto weights_info = BITensorInfo(
            weights_shape, 1, BIDataType::QSYMM8_PER_CHANNEL, weights_qinfo);
        weights.allocator()->init(weights_info);

        // 3. 创建中间输出tensor (S32)
        BITensorShape output_shape(N, M); // [M, N]
        BITensor output_s32;
        BIQuantizationInfo output_qinfo = BIQuantizationInfo(output_scale, output_offset);
        auto output_info = BITensorInfo(output_shape, 1, BIDataType::QASYMM8_SIGNED, output_qinfo);
        output_s32.allocator()->init(output_info);

        BITensorShape bias_shape(N, M);
        BITensor bias_s32;
        bias_s32.allocator()->init(BITensorInfo(bias_shape, 1, BIDataType::S32));

        // 4. 分配内存
        input.allocator()->allocate();
        weights.allocator()->allocate();
        output_s32.allocator()->allocate();
        bias_s32.allocator()->allocate();

        // 5. 拷贝数据
        std::memcpy(input.buffer(), input_data, input.info()->total_size());
        std::memcpy(weights.buffer(), weights_data, weights.info()->total_size());
        std::memcpy(bias_s32.buffer(), bias_data, bias_s32.info()->total_size());
        print_tensor(input, "input");
        print_tensor(weights, "weights");
        print_tensor(bias_s32, "bias");

        BIGEMMLowpOutputStageInfo output_stage_info;
        output_stage_info.type = BIGEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT;
        output_stage_info.output_data_type = BIDataType::QASYMM8_SIGNED; // 设置输c出数据类型
        output_stage_info.is_quantized_per_channel = true; // 因为权重是per-channel量化// 设置输出范围
        output_stage_info.gemmlowp_offset = output_offset;
        auto [min_val, max_val] = quantization::get_min_max_values_from_quantized_data_type(BIDataType::QASYMM8_SIGNED);
        output_stage_info.gemmlowp_min_bound = min_val; // 通常是-128
        output_stage_info.gemmlowp_max_bound = max_val; // 通常是127// 假设已有输入tensor的量化参数
        // 使用calculate_quantized_multipliers计算每个通道的multiplier和shift
        // 这个函数会自动填充gemmlowp_multipliers和gemmlowp_shifts
        quantization::calculate_quantized_multipliers(input_qinfo,
                                                      weights_qinfo,
                                                      output_qinfo,
                                                      output_stage_info);
        GEMMInfo gemm_info = GEMMInfo(false,
                                      false,
                                      false,
                                      2,
                                      false,
                                      false,
                                      output_stage_info,
                                      false, false, false,
                                      BIActivationLayerInfo(), false, BIWeightFormat::UNSPECIFIED, false);
        // // 9. 创建并配置输出stage
        // output_stage.configure(&output_s32, nullptr, &final_output, output_stage_info); // 10. 执行计算
        // 6. 创建输出GEMMLowp核心计算
        BINEGEMMLowpMatrixMultipleCore gemmlowp_mm_score;
        gemmlowp_mm_score.configure(&input, &weights, &bias_s32, &output_s32, gemm_info);
        gemmlowp_mm_score.run();
        // 再次进行反量化
        BITensor dq_dst0;
        dq_dst0.allocator()->init(BITensorInfo(BITensorShape(N, M), 1, BIDataType::F32));
        dq_dst0.allocator()->allocate();
        BINEDequantizationLayer dq0;
        dq0.configure(&output_s32, &dq_dst0);
        dq0.run();
        print_tensor(output_s32, "output");
        print_tensor(dq_dst0, "dq_dst0");
    }

    void print_new_tensor(const BITensor &tensor) {
        BIIOFormatInfo format;
        format.element_delim = ", "; // 元素之间用逗号分隔
        format.row_delim = "\n"; // 每行换行
        format.align_columns = 1; // 对齐列

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

    void create_input_tensor(BIITensor &tensor, const int hidden_size) {
        std::vector<float16_t> input_data(768 * hidden_size);
        // 初始化输入数据（模拟正态分布）
        for (int i = 0; i < (768 * hidden_size); ++i)
            input_data[i] = static_cast<float16_t>(((i % 32 - 16.0f) / 8.0f));

        auto *src_ptr = reinterpret_cast<float16_t *>(tensor.buffer());
        std::memcpy(src_ptr, input_data.data(), input_data.size() * sizeof(float16_t));
    }
}

TEST(GEMMLOWPCOMPARE, BASICGEMMLOWP) {
    // 1. 先给出原始值(输入先量化)
    const std::vector<int8_t> input_vec = {-49, -17, 16, 94, 127, -128};
    const std::vector<int8_t> weighs_vec = {-17, -16, 17, 52, 127, -127};
    const std::vector bias_vec = {-554, 259, -554, 259};
    constexpr float input_scale = 0.015294118021048752f;
    constexpr int input_offset = -16;
    const std::vector<float> weights_scale = {0.0118, 0.0252};
    constexpr int output_offset = -60;
    constexpr float output_scale = 0.0469412f;
    QATTest::gemmlowp_mixed_quantized(input_vec.data(),
                                      weighs_vec.data(),
                                      bias_vec.data(),
                                      input_scale,
                                      input_offset,
                                      weights_scale,
                                      output_scale,
                                      output_offset,
                                      2,
                                      3,
                                      2, 2);
}

TEST(LOWPATTETION, ATTENTIONTEST) {
    // 输入张量
    const BITensorShape input_shape(768, // sequence
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
    const BITensorShape weights_shape(2304, // input_size (width, 匹配input宽度)
                                      768); // hidden_units (height)
    const BITensorInfo weights_info(weights_shape, 1, BIDataType::QSYMM8_PER_CHANNEL, BIQuantizationInfo());
    BITensor weights;
    weights.allocator()->init(weights_info);

    // 偏置矩阵
    const BITensorShape bias_shape(2304); // hidden_units
    const BITensorInfo bias_info(bias_shape, 1, BIDataType::S32);
    BITensor bias;
    bias.allocator()->init(bias_info);

    // 权重张量
    const BITensorShape weights_shape2(768, // input_size (width, 匹配input宽度)
                                       768); // hidden_units (height)
    const BITensorInfo weights_info2(weights_shape2, 1, BIDataType::F16);
    BITensor weights2;
    weights2.allocator()->init(weights_info2);

    // 偏置矩阵
    const BITensorShape bias_shape2(768); // hidden_units
    const BITensorInfo bias_info2(bias_shape2, 1, BIDataType::F16);
    BITensor bias2;
    bias2.allocator()->init(bias_info2);

    // 输出张量
    const BITensorShape output_shape(768, // hidden_units (width)
                                     16,
                                     5); // batch_size (height)
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
    QATTest::create_input_tensor(input, 2);
    QATTest::fill_new_tensor_val(weights, static_cast<float16_t>(1));
    QATTest::fill_new_tensor_val(bias, static_cast<float16_t>(1));

    QATTest::fill_new_tensor_val(add_tensor, static_cast<float16_t>(1));
    QATTest::fill_new_tensor_val(weights2, static_cast<float16_t>(1));
    QATTest::fill_new_tensor_val(bias2, static_cast<float16_t>(1));
    QATTest::fill_new_tensor_val(gamma, static_cast<float16_t>(1));

    auto scalar_ptr = reinterpret_cast<float16_t *>(scalar.buffer());
    scalar_ptr[0] = 0.5f;

    BINEAttentionLowpLayer attention_layer;
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
}

TEST(QUANTIZE_TEST, DEQUAN_EXAM) {
    // 1. 获取已经量化的结果
    BITensor input, output;
    std::vector<int8_t> input_data{-128, -91, -55, 127};
    BITensorShape input_shape(4);
    BITensorInfo input_info{
        input_shape,
        1,
        BIDataType::QASYMM8_SIGNED,
        BIQuantizationInfo(0.054901960784313725, -55)
    };
    input.allocator()->init(input_info);
    input.allocator()->allocate();
    std::memcpy(input.buffer(), input_data.data(), input.info()->total_size());
    QATTest::print_tensor(input, "input");
    output.allocator()->init(BITensorInfo(input_shape, 1, BIDataType::F16));
    output.allocator()->allocate();

    // 2. 进行反量化运算
    BINEDequantizationLayer dequantization_layer;
    dequantization_layer.configure(&input, &output);
    dequantization_layer.run();

    QATTest::print_tensor(output, "output");
}

TEST(QUANTIZE_TE2ST, QUAN_EXAM) {
    // 1. 获取已经量化的结果
    BITensor input, output;
    std::vector<float16_t> input_data{-1.0, -0.5, 0.0, 1.2, 1.7, -2.2};
    BITensorShape input_shape(3, 2);
    BITensorInfo input_info{
        input_shape,
        1,
        BIDataType::F16
    };
    input.allocator()->init(input_info);
    input.allocator()->allocate();
    std::memcpy(input.buffer(), input_data.data(), input.info()->total_size());
    QATTest::print_tensor(input, "input");
    output.allocator()->init(BITensorInfo(input_shape, 1, BIDataType::QASYMM8_SIGNED,
                                          BIQuantizationInfo(0.015294118021048752, 16)));
    output.allocator()->allocate();
    // 2. 进行反量化运算
    BINEQuantizationLayer quantization_layer;
    quantization_layer.configure(&input, &output);
    quantization_layer.run();

    QATTest::print_tensor(output, "output");
}

TEST(QUANTIZE_TEST, DEQUANT) {
    // 1. 获取量化的per channel
    const std::vector<int8_t> weights_vec = {-17, -16, 17, 52, 127, -127};
    const std::vector<float> weights_scale = {
        0.0118, 0.0252
    };
    BIQuantizationInfo weight_info = BIQuantizationInfo(weights_scale);
    BITensorShape weights_shape(2, 3); // [K, N]
    BITensor weights;
    auto weights_info = BITensorInfo(
        weights_shape, 1, BIDataType::QSYMM8_PER_CHANNEL, weight_info);
    weights.allocator()->init(weights_info);
    weights.allocator()->allocate();
    std::memcpy(weights.buffer(), weights_vec.data(), weights.info()->total_size());
    BITensor output;
    BITensorInfo input_info{
        weights_shape,
        1,
        BIDataType::F16
    };
    output.allocator()->init(input_info);
    output.allocator()->allocate();
    // 2. 进行反量化
    BINEDequantizationLayer dequantization_layer;
    dequantization_layer.configure(&weights, &output);
    dequantization_layer.run();
    QATTest::print_tensor(output, "weights");
}

