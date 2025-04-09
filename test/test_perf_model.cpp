//
// Created by Mason on 2025/2/17.
//

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <runtime/bi_tensor.hpp>
#include <runtime/neon/bi_ne_functions.h>
#include <utils/utils.hpp>
#include <runtime/bi_scheduler.hpp>
#include <thread>

#include "data/core/utils/quantization/asymm_helpers.hpp"

using namespace BatmanInfer;

namespace PerfTest {
    void print_tensor_shape(const BITensor &tensor) {
        int dims = tensor.info()->num_dimensions();
        for (int i = 0; i < dims; i++) {
            std::cout << tensor.info()->dimension(i) << " ";
        }
        std::cout << std::endl;
    }

    void invert_qinfo_offset(BITensor &t) {
        BIQuantizationInfo qinfo = t.info()->quantization_info();
        t.info()->set_quantization_info(BIQuantizationInfo(qinfo.scale()[0], -qinfo.offset()[0], qinfo.is_dynamic()));
    }

    void print_tensor(const BatmanInfer::BITensor &tensor,
                      const std::string &name = "temp",
                      const BatmanInfer::BIIOFormatInfo::PrintRegion region =
                              BIIOFormatInfo::PrintRegion::Full) {
        std::cout << name << std::endl;
        BatmanInfer::BIIOFormatInfo format;
        format.element_delim = ", "; // 元素之间用逗号分隔
        format.row_delim = "\n"; // 每行换行
        format.align_columns = 1; // 对齐列
        format.print_region = region;

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

    BIGEMMLowpOutputStageInfo configure_gemm_output_stage(float input_scale,
                                                          const std::vector<float> &weight_scales,
                                                          float output_scale,
                                                          int32_t output_offset) {
        // 1. 创建output stage配置
        BIGEMMLowpOutputStageInfo output_stage;
        output_stage.type = BIGEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT;
        output_stage.output_data_type = BIDataType::QASYMM8_SIGNED;

        // 2. 计算每个通道的combined scale
        const int N = weight_scales.size();
        std::vector<int32_t> multipliers(N);
        std::vector<int32_t> shifts(N);
        auto [min_val, max_val] = quantization::get_min_max_values_from_quantized_data_type(BIDataType::QASYMM8_SIGNED);
        output_stage.gemmlowp_min_bound = min_val; // 通常是-128
        output_stage.gemmlowp_max_bound = max_val; // 通常是127// 假设已有输入tensor的量化参数

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
        // BINEGEMMLowpOutputStage output_stage_kernel;
        // output_stage_kernel.configure(input, nullptr, output, output_stage);
        return output_stage;
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
        const std::string &input_str,
        const std::string &weight_str,
        const std::string &bias_str,
        const std::string &second_i_str,
        float input_scale,
        int input_offset,
        const std::vector<float> &weight_scales,
        float output_scale,
        int output_offset) {
        // 1. 创建输入tensor (QASYMM8)
        int B = 1;
        int M = 1, K = 768, N = 2304;
        BIQuantizationInfo input_qinfo, weights_qinfo;
        auto input = create_qasymm8(input_scale,
                                    input_offset,
                                    std::vector<int>{B, M, K},
                                    input_qinfo,
                                    input_str);

        // print_tensor(input, "input");


        // 2. 创建权重tensor (QSYMM8_PER_CHANNEL)
        auto weight = create_per_channel(weight_scales,
                                         std::vector<int>{K, N},
                                         weights_qinfo,
                                         weight_str);
        //        print_tensor(weight, "weight");

        // 3. 创建中间输出tensor (S32)
        BITensorShape output_shape(N, M, B); // [M, N]
        BITensor s32_output;
        s32_output.allocator()->init(BITensorInfo(output_shape, 1, BIDataType::S32));
        s32_output.allocator()->allocate();
        // // 9. 创建并配置输出stage
        // output_stage.configure(&output_s32, nullptr, &final_output, output_stage_info); // 10. 执行计算
        // 6. 创建输出GEMMLowp核心计算
        BINEGEMMLowpMatrixMultipleCore gemmlowp_mm_score;
        gemmlowp_mm_score.configure(&input, &weight, nullptr, &s32_output);
        gemmlowp_mm_score.run();

        BITensor int8_output;
        BIQuantizationInfo output_qinfo = BIQuantizationInfo(output_scale, output_offset);
        auto output_info = BITensorInfo(output_shape, 1, BIDataType::QASYMM8_SIGNED, output_qinfo);
        int8_output.allocator()->init(output_info);
        int8_output.allocator()->allocate();
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
        // output_stage_info = configure_gemm_output_stage(input_scale, weight_scales, output_scale, output_offset);
        quantization::calculate_quantized_multipliers(input_qinfo,
                                                      weights_qinfo,
                                                      output_qinfo,
                                                      output_stage_info);


        BINEGEMMLowpOutputStage output_stage;
        output_stage.configure(&s32_output, nullptr, &int8_output, output_stage_info);
        output_stage.run();

        BITensor fp16_output;
        auto fp16_info = BITensorInfo(output_shape, 1, BIDataType::F16);
        fp16_output.allocator()->init(fp16_info);
        fp16_output.allocator()->allocate();
        BINEDequantizationLayer dequantization_layer;
        dequantization_layer.configure(&int8_output, &fp16_output);
        dequantization_layer.run();

        print_tensor(fp16_output, "output");

        // input.allocator()->free();
        //
        // B = 10;
        // M = 16;
        //
        // input = create_qasymm8(input_scale,
        //                        input_offset,
        //                        std::vector<int>{B, M, K},
        //                        input_qinfo,
        //                        second_i_str);
        // output_s32.allocator()->free();
        //
        // output_shape = BITensorShape(N, M, B);
        // output_info = BITensorInfo(output_shape, 1, BIDataType::QASYMM8_SIGNED, output_qinfo);
        // output_s32.allocator()->init(output_info);
        // output_s32.allocator()->allocate();
        //
        // gemmlowp_mm_score.run();
        //
        // print_tensor(output_s32, "output_2");

        // 修改输入量化值

        // 再次进行反量化
        //        BITensor dq_dst0;
        //        dq_dst0.allocator()->init(BITensorInfo(BITensorShape(N, M), 1, BIDataType::F32));
        //        dq_dst0.allocator()->allocate();
        //        BINEDequantizationLayer dq0;
        //        dq0.configure(&output_s32, &dq_dst0);
        //        dq0.run();
        //        print_tensor(output_s32, "output");
        //        print_tensor(dq_dst0, "dq_dst0");
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

    template<typename T>
    void fill_from_one(const BITensor &tensor) {
        auto tensor_ptr = reinterpret_cast<T *>(tensor.buffer());
        size_t num_elements = tensor.info()->tensor_shape().total_size();
        // 获取元素数量
        for (size_t i = 0; i < num_elements; ++i) {
            tensor_ptr[i] = static_cast<T>(i) * 0.00001;
        }
    }

    template<typename T>
    void fill_tensor_val_with_arr(const BITensor &tensor, const std::vector<T> val) {
        auto tensor_ptr = reinterpret_cast<T *>(tensor.buffer());
        size_t num_elements = tensor.info()->tensor_shape().total_size(); // 获取元素数量
        for (size_t i = 0; i < num_elements; ++i) {
            tensor_ptr[i] = val[i];
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

    template<typename T>
    void print_special_tensor(BITensor &tensor) {
        auto total_len = tensor.info()->tensor_shape().total_size();
        auto data_ptr = reinterpret_cast<uint8_t *>(tensor.buffer());
        for (int i = 0; i < total_len; i++) {
            std::cout << static_cast<int>(data_ptr[i]) << " ";
        }
    }
}

template<typename T>
void fill_model_tensor_val(const BITensor &tensor, const T val) {
    auto tensor_ptr = reinterpret_cast<T *>(tensor.buffer());
    size_t num_elements = tensor.info()->tensor_shape().total_size(); // 获取元素数量
    for (size_t i = 0; i < num_elements; ++i) {
        tensor_ptr[i] = val;
    }
}

/**
 * 性能统计结构体
 */
struct PerfStats {
    double avg_ms; // 平均耗时
    double std_dev_ms; // 标准差
    double min_ms; // 最小耗时
    double max_ms; // 最大耗时
    size_t iterations; // 有效迭代次数
};

// 定义成员函数指针类型
using MemberFunc = void (BIIFunction::*)();

//PerfStats measure_performance(BIIFunction *obj,
//                              MemberFunc kernel_func,
//                              size_t warmup = 10,
//                              size_t iterations = 1000,
//                              double outlier_threshold = 3.0) {
//
//
//}

TEST(ModelPerfTest, GPT2Perf) {
    //    BIScheduler::set(BIScheduler::Type::OMP);
    BIScheduler::get().set_num_threads(std::thread::hardware_concurrency());
    BIMemoryGroup group{BIMemoryManagerOnDemand::make_default()};
    // 先确定需要的算子
    BINEAttentionLayer attention_layer;
    BINEArithmeticAddition add_f;
    BINEFeedForwardLayer feedforward_layer;
    BINEArithmeticAddition add_2_f;

    // 输入张量
    const BITensorShape input_shape(768, // hidden size
                                    16, // sequence length
                                    5); // batch size
    const BITensorShape gamma_shape(768);
    const BITensorShape fc_weights_shape(3072, // input_size (width, 匹配input宽度)
                                         768); // hidden_units (height)
    const BITensorShape fc_bias_shape(3072); // hidden_units
    // 权重张量
    const BITensorShape proj_weights_shape2(768, // input_size (width, 匹配input宽度)
                                            3072); // hidden_units (height)
    const BITensorShape proj_bias_shape2(768); // hidden_units

    const BITensorShape output_shape(768, // hidden_units (width)
                                     16,
                                     5); // batch_size (height)
    const BIActivationLayerInfo act_info(BIActivationFunction::GELU);

    // 权重张量
    const BITensorShape weights_shape(2304, // input_size (width, 匹配input宽度)
                                      768); // hidden_units (height)

    // 偏置矩阵
    const BITensorShape bias_shape(2304); // hidden_units

    // 权重张量
    const BITensorShape weights_shape2(768, // input_size (width, 匹配input宽度)
                                       768); // hidden_units (height)

    // 偏置矩阵
    const BITensorShape bias_shape2(768); // hidden_units

    // 标量
    const BITensorShape scalar_shape(1);

    // 相加权重
    const BITensorShape add_shape(16, 16);

    PermutationVector perm{0, 2, 1, 3};
    PermutationVector perm2{2, 0, 1, 3};
    PermutationVector perm_final{0, 2, 1, 3};

    auto input = utils::create_tensor(input_shape, nullptr);
    const auto gamma = utils::create_npy_tensor("./input_res/rms_attention_1.npy", gamma_shape);
    const auto fc_weights = utils::create_npy_tensor("./input_res/mlp_c_fc_weight.npy",
                                                     fc_weights_shape);
    const auto fc_bias = utils::create_npy_tensor("./input_res/mlp_c_fc_bias.npy", fc_bias_shape);
    const auto proj_weights = utils::create_npy_tensor("./input_res/mlp_c_proj_weight.npy",
                                                       proj_weights_shape2);
    const auto proj_bias = utils::create_npy_tensor("./input_res/mlp_c_proj_bias.npy",
                                                    proj_bias_shape2);
    auto output = utils::create_tensor(output_shape, nullptr);
    const auto weights = utils::create_npy_tensor("./input_res/attn_c_attn_weight.npy",
                                                  weights_shape);
    const auto bias = utils::create_npy_tensor("./input_res/attn_c_attn_bias.npy", bias_shape);
    const auto weights2 = utils::create_npy_tensor("./input_res/attn_c_proj_weight_2.npy",
                                                   weights_shape2);
    const auto bias2 = utils::create_npy_tensor("./input_res/attn_c_proj_bias_2.npy", bias_shape2);
    const auto gamma2 = utils::create_npy_tensor("./input_res/mlp_ln_2_weight.npy", gamma_shape);
    const auto scalar = utils::create_tensor(scalar_shape, nullptr);
    const auto add_tensor = utils::create_npy_tensor("./input_res/_attn_Where_output_0.npy", add_shape);

    // 加法结果
    auto add_temp_out = utils::create_tensor(input_shape, nullptr);
    auto ffn_out = utils::create_tensor(input_shape, nullptr);
    auto final_out = utils::create_tensor(input_shape, nullptr);

    fill_model_tensor_val(scalar, static_cast<float16_t>(0.3535533845424652));


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

    add_f.configure(&output, &input, &add_temp_out, BIConvertPolicy::WRAP);

    feedforward_layer.configure(&add_temp_out, &fc_weights,
                                &fc_bias,
                                &proj_weights,
                                &proj_bias,
                                &gamma2,
                                act_info,
                                &ffn_out,
                                5,
                                16);

    add_2_f.configure(&add_temp_out, &ffn_out, &final_out, BIConvertPolicy::WRAP);

    const auto warmup = 10; // 预热次数
    const auto iterations = 1000; // 运行次数
    const double outlier_threshold = 3.0; // 异常值阈值(标准差倍数)

    std::vector<double> timings;
    timings.reserve(iterations);

    // 预测阶段（不记录时间）
    for (size_t i = 0; i < warmup; ++i) {
        std::vector<float16_t> input_data(768 * 16);
        for (int i = 0; i < 768 * 16; i++) {
            input_data[i] = static_cast<float16_t>(i + 1) / 1000;
        }
        std::memcpy(input.buffer(), input_data.data(), 768 * 16 * sizeof(float16_t));
        attention_layer.run();
        add_f.run();
        feedforward_layer.run();
        add_2_f.run();
    }

    // 修改input的sequence长度

    // 正式测量
    for (size_t i = 0; i < iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        attention_layer.run();
        add_f.run();
        feedforward_layer.run();
        add_2_f.run();
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

    auto perf_status = PerfStats{
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

    return;
}

TEST(ModelPerfTest, KVCaches) {
    // 1. 预分配缓存空间（示例为最大长度128）
    size_t head_dim = 4;
    size_t num_heads = 2;
    BITensor k_cache;
    BITensorInfo cache_info(BITensorShape(head_dim, num_heads, 128), 1, BIDataType::F16);
}

TEST(ModelPerfTest, MLPOriginPerf) {
    BIScheduler::set(BIScheduler::Type::OMP);
    BIScheduler::get().set_num_threads(std::thread::hardware_concurrency());
    BIMemoryGroup group{BIMemoryManagerOnDemand::make_default()};
    // 先确定需要的算子
    BINEAttentionLayer attention_layer;
    BINEArithmeticAddition add_f;
    BINEFeedForwardLayer feedforward_layer;
    BINEArithmeticAddition add_2_f;

    // 输入张量
    const BITensorShape input_shape(768, // hidden size
                                    16, // sequence length
                                    5); // batch size
    const BITensorShape gamma_shape(768);
    const BITensorShape fc_weights_shape(3072, // input_size (width, 匹配input宽度)
                                         768); // hidden_units (height)
    const BITensorShape fc_bias_shape(3072); // hidden_units
    // 权重张量
    const BITensorShape proj_weights_shape2(768, // input_size (width, 匹配input宽度)
                                            3072); // hidden_units (height)
    const BITensorShape proj_bias_shape2(768); // hidden_units

    const BITensorShape output_shape(768, // hidden_units (width)
                                     16,
                                     5); // batch_size (height)
    const BIActivationLayerInfo act_info(BIActivationFunction::GELU);

    // 权重张量
    const BITensorShape weights_shape(2304, // input_size (width, 匹配input宽度)
                                      768); // hidden_units (height)

    // 偏置矩阵
    const BITensorShape bias_shape(2304); // hidden_units

    // 权重张量
    const BITensorShape weights_shape2(768, // input_size (width, 匹配input宽度)
                                       768); // hidden_units (height)

    // 偏置矩阵
    const BITensorShape bias_shape2(768); // hidden_units

    // 标量
    const BITensorShape scalar_shape(1);

    // 相加权重
    const BITensorShape add_shape(16, 16);

    PermutationVector perm{0, 2, 1, 3};
    PermutationVector perm2{2, 0, 1, 3};
    PermutationVector perm_final{0, 2, 1, 3};

    auto input = utils::create_tensor(input_shape, nullptr);
    const auto gamma = utils::create_npy_tensor("./input_res/rms_attention_1.npy", gamma_shape);
    const auto fc_weights = utils::create_npy_tensor("./input_res/mlp_c_fc_weight.npy",
                                                     fc_weights_shape);
    const auto fc_bias = utils::create_npy_tensor("./input_res/mlp_c_fc_bias.npy", fc_bias_shape);
    const auto proj_weights = utils::create_npy_tensor("./input_res/mlp_c_proj_weight.npy",
                                                       proj_weights_shape2);
    const auto proj_bias = utils::create_npy_tensor("./input_res/mlp_c_proj_bias.npy",
                                                    proj_bias_shape2);
    auto output = utils::create_tensor(output_shape, nullptr);
    const auto weights = utils::create_npy_tensor("./input_res/attn_c_attn_weight.npy",
                                                  weights_shape);
    const auto bias = utils::create_npy_tensor("./input_res/attn_c_attn_bias.npy", bias_shape);
    const auto weights2 = utils::create_npy_tensor("./input_res/attn_c_proj_weight_2.npy",
                                                   weights_shape2);
    const auto bias2 = utils::create_npy_tensor("./input_res/attn_c_proj_bias_2.npy", bias_shape2);
    const auto gamma2 = utils::create_npy_tensor("./input_res/mlp_ln_2_weight.npy", gamma_shape);
    const auto scalar = utils::create_tensor(scalar_shape, nullptr);
    const auto add_tensor = utils::create_npy_tensor("./input_res/_attn_Where_output_0.npy", add_shape);

    // 加法结果
    auto add_temp_out = utils::create_tensor(input_shape, nullptr);
    auto ffn_out = utils::create_tensor(input_shape, nullptr);
    auto final_out = utils::create_tensor(input_shape, nullptr);

    fill_model_tensor_val(scalar, static_cast<float16_t>(0.3535533845424652));


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

    add_f.configure(&output, &input, &add_temp_out, BIConvertPolicy::WRAP);

    feedforward_layer.configure(&add_temp_out, &fc_weights,
                                &fc_bias,
                                &proj_weights,
                                &proj_bias,
                                &gamma2,
                                act_info,
                                &ffn_out,
                                5,
                                16);

    add_2_f.configure(&add_temp_out, &ffn_out, &final_out, BIConvertPolicy::WRAP);

    const auto warmup = 10; // 预热次数
    const auto iterations = 1000; // 运行次数
    const double outlier_threshold = 3.0; // 异常值阈值(标准差倍数)

    std::vector<double> timings;
    timings.reserve(iterations);

    // 预测阶段（不记录时间）
    for (size_t i = 0; i < warmup; ++i) {
        std::vector<float16_t> input_data(768 * 16);
        for (int i = 0; i < 768 * 16; i++) {
            input_data[i] = static_cast<float16_t>(i + 1) / 1000;
        }
        std::memcpy(input.buffer(), input_data.data(), 768 * 16 * sizeof(float16_t));
        attention_layer.run();
        add_f.run();
        feedforward_layer.run();
        // add_2_f.run();
    }

    // 修改input的sequence长度

    // 正式测量
    for (size_t i = 0; i < iterations; ++i) {
        attention_layer.run();
        add_f.run();
        auto start = std::chrono::high_resolution_clock::now();
        feedforward_layer.run();
        auto end = std::chrono::high_resolution_clock::now();
        // add_2_f.run();

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

    auto perf_status = PerfStats{
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

TEST(AnotherTest, NewMLPOriginPerf) {
    BIScheduler::set(BIScheduler::Type::OMP);
    BIScheduler::get().set_num_threads(std::thread::hardware_concurrency());
    BIMemoryGroup group{BIMemoryManagerOnDemand::make_default()};
    const int batch_size = 5;
    const int seq_len = 16;
    // 1. 先初始化输入矩阵
    const std::string &input_path = "./input_res_1/mlp_input.npy";
    BITensor input = PerfTest::create_norm_input(std::vector<int>{batch_size, seq_len, 768}, input_path);
    PerfTest::print_tensor(input, "input");
    // 2. 初始化gamma张量
    const std::string &gamma_path = "./input_res_1/mlp_rms_gamma.npy";
    BITensor gamma = PerfTest::create_norm_input(std::vector<int>{768}, gamma_path);
    // 3. 初始化fc_weights的权重
    const std::string &c_fc_weights_path =
            "./input_res_1/reordered_c_fc_weights.npy";
    std::vector<float> c_fc_weights_scales;
    // 量化信息
    BIQuantizationInfo c_fc_weight_qinfo;
    std::ifstream c_fc_weights_scale_file(
        "./input_res_1/c_fc_scales.txt");
    float value;

    while (c_fc_weights_scale_file >> value) {
        c_fc_weights_scales.push_back(value);
    }
    BITensor c_fc_weights = PerfTest::create_per_channel(c_fc_weights_scales, std::vector{768, 3072},
                                                         c_fc_weight_qinfo, c_fc_weights_path);
    // QATTest::print_tensor(c_fc_weights, "c_fc_weights");
    // 4. 初始化fc_bias
    const std::string &c_fc_bias_path = "./input_res_1/c_fc_bias.npy";
    BITensor c_fc_bias = PerfTest::create_norm_bias(3072, c_fc_bias_path);
    PerfTest::print_tensor(c_fc_bias, "c_fc_bias");
    // 5. 输出张量
    BITensor output;
    output.allocator()->init(BITensorInfo(BITensorShape(768, seq_len, batch_size), 1, BIDataType::F16));
    output.allocator()->allocate();

    // 6. proj的权重
    const std::string &c_proj_path = "./input_res_1/c_proj_weights.npy";
    BITensor c_proj_weight = PerfTest::create_norm_input(std::vector<int>{3072, 768}, c_proj_path);
    // QATTest::print_tensor(c_proj, "c_proj");

    const std::string &c_proj_bias_path =
            "./input_res_1/c_proj_bias.npy";
    BITensor c_proj_bias = PerfTest::create_norm_input(std::vector<int>{768}, c_proj_bias_path);
    // QATTest::print_tensor(c_proj_bias, "c_proj_bias");

    BINEMLPLayer _mlp_layer;
    float fc1_input_scale = 0.006902442025203331f;
    int fc1_input_zero_point = -9;
    float fc1_output_scale = 0.1969725440530216f;
    int fc1_output_zero_point = -19;
    float gelu_output_scale = 0.11368115240452337f;
    int gelu_output_zero_point = -127;
    _mlp_layer.configure(&input, fc1_input_scale,
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
                         &output,
                         batch_size,
                         seq_len);
    const auto warmup = 10; // 预热次数
    const auto iterations = 1000; // 运行次数
    const double outlier_threshold = 3.0; // 异常值阈值(标准差倍数)
    std::vector<double> timings;
    timings.reserve(iterations);

    // 预测阶段（不记录时间）
    for (size_t i = 0; i < warmup; ++i) {
        _mlp_layer.run();
    } // 正式测量
    for (size_t i = 0; i < iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        _mlp_layer.run();
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

    auto perf_status = PerfStats{
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

