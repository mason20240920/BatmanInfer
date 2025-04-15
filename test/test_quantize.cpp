//
// Created by Mason on 2025/3/11.
//
#include <runtime/neon/bi_ne_functions.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <fstream>

#include "data/core/utils/quantization/asymm_helpers.hpp"
#include "utils/utils.hpp"
#include <runtime/bi_scheduler.hpp>
#include <thread>
#include <limits>
#include "function_info/bi_MatMulInfo.h"

using namespace BatmanInfer;

namespace QATTest {
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
        int B = 2;
        int M = 2, K = 768, N = 2304;
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
        print_tensor(s32_output, "output");

        // BITensor fp16_output;
        // auto fp16_info = BITensorInfo(output_shape, 1, BIDataType::F16);
        // fp16_output.allocator()->init(fp16_info);
        // fp16_output.allocator()->allocate();
        // BINEDequantizationLayer dequantization_layer;
        // dequantization_layer.configure(&int8_output, &fp16_output);
        // dequantization_layer.run();
        //
        // print_tensor(fp16_output, "output");

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
        size_t num_elements = tensor.info()->tensor_shape().total_size() / tensor.info()->element_size(); // 获取元素数量
        for (size_t i = 0; i < num_elements; ++i) {
            tensor_ptr[i] = val;
        }
    }

    template<typename T>
    void fill_from_one(const BITensor &tensor) {
        auto tensor_ptr = reinterpret_cast<T *>(tensor.buffer());
        size_t num_elements = tensor.info()->tensor_shape().total_size() / tensor.info()->element_size();
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

TEST(GEMMLOWPCOMPARE, BASICGEMMLOWP) {
    // BIScheduler::set(BIScheduler::Type::OMP);
    BIScheduler::get().set_num_threads(std::thread::hardware_concurrency());
    // 1. 先给出原始值(输入先量化)
    const std::string &input_path = "/Users/mason/Desktop/Desktop/PythonProjects/gemmlowp_compare/input.npy";
    const std::string &weight_path = "/Users/mason/Desktop/Desktop/PythonProjects/gemmlowp_compare/weights.npy";
    const std::string &second_i_path = "/Users/mason/Desktop/Desktop/PythonProjects/gemmlowp_compare/input_second.npy";
    const std::string &bias_path = "";
    constexpr float input_scale = 0.011764705882352941f;
    constexpr int input_offset = 43;
    // 读取权重的scales
    std::vector<float> weight_scales;
    std::ifstream in_file("/Users/mason/Desktop/Desktop/PythonProjects/gemmlowp_compare/weight_scale.txt");
    float value;

    while (in_file >> value) {
        weight_scales.push_back(value);
    }
    constexpr int output_offset = -6;
    constexpr float output_scale = 0.14801221361347272f;
    QATTest::gemmlowp_mixed_quantized(input_path,
                                      weight_path,
                                      bias_path,
                                      second_i_path,
                                      input_scale,
                                      input_offset,
                                      weight_scales,
                                      output_scale,
                                      output_offset);
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

TEST(QUANTIZE_MATMUL, MATMULQ) {
    const std::string &a_path = "/Users/mason/Desktop/Desktop/PythonProjects/gemmlowp_compare/a.npy";
    const std::string &a_2_path = "/Users/mason/Desktop/Desktop/PythonProjects/gemmlowp_compare/a_2.npy";
    const std::string &b_path = "/Users/mason/Desktop/Desktop/PythonProjects/gemmlowp_compare/b.npy";
    const std::string &b_2_path = "/Users/mason/Desktop/Desktop/PythonProjects/gemmlowp_compare/b_2.npy";
    int B = 1, S = 1, M = 16, K = 64, N = 16;
    float a_scale = 0.011764705882352941f;
    int8_t a_zp = -43;

    // 量化的A矩阵
    BIQuantizationInfo a_qinfo, b_qinfo;
    auto a_tensor = QATTest::create_qasymm8(a_scale,
                                            a_zp,
                                            std::vector<int>{B, S, M, K},
                                            a_qinfo,
                                            a_path);
    auto b_tensor = QATTest::create_qasymm8(a_scale,
                                            a_zp,
                                            std::vector<int>{B, S, K, N},
                                            a_qinfo,
                                            b_path);
    a_tensor.info()->set_are_values_constant(false);
    b_tensor.info()->set_are_values_constant(false);
    // QATTest::print_tensor(a_tensor, "a_tensor");


    // 3. 创建中间输出tensor (S32)
    BITensorShape output_shape(N, M, S, B); // [M, N]
    BITensor output_s32;
    float output_scale =
            0.15568818391538133f;
    int output_offset = -103;
    BIQuantizationInfo output_qinfo = BIQuantizationInfo(output_scale, output_offset);
    auto output_info = BITensorInfo(output_shape, 1, BIDataType::QASYMM8_SIGNED, output_qinfo);
    output_s32.allocator()->init(output_info);
    output_s32.allocator()->allocate();

    // 定义 MatMul 配置信息
    BIMatMulInfo matmul_info; // 不转置左矩阵，转置右矩阵
    matmul_info.adj_lhs(false).adj_rhs(false);
    BICpuMatMulSettings settings;
    settings.fast_math(true); // 启用快速数学模式

    BINEMatMul matmul;
    matmul.configure(&a_tensor, &b_tensor, &output_s32, matmul_info, settings);
    matmul.run();

    QATTest::print_tensor(output_s32, "output_s32");

    a_tensor.allocator()->free();
    b_tensor.allocator()->free();
    output_s32.allocator()->free();

    B = 10, S = 16;
    a_tensor = QATTest::create_qasymm8(a_scale,
                                       a_zp,
                                       std::vector<int>{B, S, M, K},
                                       a_qinfo,
                                       a_2_path);
    b_tensor = QATTest::create_qasymm8(a_scale,
                                       a_zp,
                                       std::vector<int>{B, S, K, N},
                                       a_qinfo,
                                       b_2_path);
    output_shape = BITensorShape(N, M, S, B); // [M, N]
    output_info = BITensorInfo(output_shape, 1, BIDataType::QASYMM8_SIGNED, output_qinfo);
    output_s32.allocator()->init(output_info);
    output_s32.allocator()->allocate();
    a_tensor.info()->set_are_values_constant(false);
    b_tensor.info()->set_are_values_constant(false);
    matmul.dynamic_configure(&a_tensor, &b_tensor, &output_s32);
    //    matmul.configure(&a_tensor, &b_tensor, &output_s32, matmul_info, settings);
    matmul.run();

    QATTest::print_tensor(output_s32, "output_s32");
    // matmul.configure(&a_tensor, &b_tensor, &output_s32, matmul_info, settings);
}

TEST(DYNAMIC_KERNEL, KERNEL_SIZE_PICK) {
    // 目的查看目前内核的选择的选择图
    // 范围从[B, S, N, M] 从 B = 1, S = 1到B = 20, S = 16
    constexpr float input_scale = 0.011764705882352941f;
    constexpr int input_offset = -43;
    int B = 1, S = 1, N = 16, K = 64, M = 16;
    for (int i = 1; i < 20; i++)
        for (int j = 1; j < 16; j++) {
            // 1. 初始化A矩阵
            B = i, S = j;
            BIQuantizationInfo quantizationInfo = BIQuantizationInfo(input_scale, input_offset);
            BITensorShape a_shape = BITensorShape(K, N, S, B);
            BITensorShape b_shape = BITensorShape(M, K, S, B);
            BITensorShape c_shape = BITensorShape(M, N, S, B);
            BITensorInfo a_info = BITensorInfo(a_shape, 1, BIDataType::QASYMM8_SIGNED, quantizationInfo);
            BITensorInfo b_info = BITensorInfo(b_shape, 1, BIDataType::QASYMM8_SIGNED, quantizationInfo);
            BITensorInfo c_info = BITensorInfo(c_shape, 1, BIDataType::QASYMM8_SIGNED, quantizationInfo);
            BITensor a, b, c;
            a.allocator()->init(a_info);
            b.allocator()->init(b_info);
            c.allocator()->init(c_info);
            a.allocator()->allocate();
            b.allocator()->allocate();
            c.allocator()->allocate();

            // 定义 MatMul 配置信息
            BIMatMulInfo matmul_info; // 不转置左矩阵，转置右矩阵
            matmul_info.adj_lhs(false).adj_rhs(false);
            BICpuMatMulSettings settings;
            settings.fast_math(true); // 启用快速数学模式
            a.info()->set_are_values_constant(false);
            b.info()->set_are_values_constant(false);

            BINEMatMul matmul;
            matmul.configure(&a, &b, &c, matmul_info, settings);
        }
}

TEST(DYNAMIC_KERNEL, KERNEL_GEMM_SIZE_PICK) {
    // 目的查看目前内核的选择的GEMM汇编函数
    // 范围从[B, S, N, M] 从 B = 1, S = 1到B = 20, S = 16
    constexpr float input_scale = 0.011764705882352941f;
    constexpr int input_offset = -43;
    int B = 1, S = 1, K = 768, M = 2304;
    std::vector<float> weight_scales;
    BIQuantizationInfo weight_qinfo;
    //    std::ifstream in_file("/data/local/tmp/onnx_runtime/build/weight_scale.txt");
    //    const std::string &weight_path = "/data/local/tmp/onnx_runtime/build/weights.npy";
    std::ifstream in_file("/Users/mason/Desktop/Desktop/PythonProjects/gemmlowp_compare/weight_scale.txt");
    const std::string &weight_path = "/Users/mason/Desktop/Desktop/PythonProjects/gemmlowp_compare/weights.npy";
    // 权重矩阵
    BITensor weight = QATTest::create_per_channel(weight_scales, std::vector<int>{K, M}, weight_qinfo, weight_path);
    for (int i = 1; i < 20; i++)
        for (int j = 1; j < 16; j++) {
            // 1. 初始化A矩阵
            B = i, S = j;
            BIQuantizationInfo quantizationInfo = BIQuantizationInfo(input_scale, input_offset);
            BITensorShape a_shape = BITensorShape(K, S, B);
            BITensorShape c_shape = BITensorShape(M, S, B);
            BITensorInfo a_info = BITensorInfo(a_shape, 1, BIDataType::QASYMM8_SIGNED, quantizationInfo);
            BITensorInfo c_info = BITensorInfo(c_shape, 1, BIDataType::QASYMM8_SIGNED, quantizationInfo);
            BITensor a, c;
            a.allocator()->init(a_info);
            c.allocator()->init(c_info);
            a.allocator()->allocate();
            c.allocator()->allocate();

            BIGEMMLowpOutputStageInfo output_stage_info;
            output_stage_info.type = BIGEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT;
            output_stage_info.output_data_type = BIDataType::QASYMM8_SIGNED; // 设置输c出数据类型
            output_stage_info.is_quantized_per_channel = true; // 因为权重是per-channel量化// 设置输出范围
            output_stage_info.gemmlowp_offset = input_offset;
            auto [min_val, max_val] = quantization::get_min_max_values_from_quantized_data_type(
                BIDataType::QASYMM8_SIGNED);
            output_stage_info.gemmlowp_min_bound = min_val; // 通常是-128
            output_stage_info.gemmlowp_max_bound = max_val; // 通常是127// 假设已有输入tensor的量化参数
            // 使用calculate_quantized_multipliers计算每个通道的multiplier和shift
            // 这个函数会自动填充gemmlowp_multipliers和gemmlowp_shifts
            quantization::calculate_quantized_multipliers(quantizationInfo,
                                                          weight_qinfo,
                                                          quantizationInfo,
                                                          output_stage_info);
            GEMMInfo gemm_info = GEMMInfo(false,
                                          false,
                                          true,
                                          false,
                                          false,
                                          false,
                                          output_stage_info,
                                          false, false, false,
                                          BIActivationLayerInfo(), false, BIWeightFormat::UNSPECIFIED, false);


            BINEGEMMLowpMatrixMultipleCore gemmlowp_mm_score;
            std::cout << "B: " << B << "\tS:" << S << std::endl;
            gemmlowp_mm_score.configure(&a, &weight, nullptr, &c, gemm_info);
        }
}

TEST(NEGatherTest, NEGatherStaticTest) {
    // 1. 先读取numpy的Gather权重
    BITensorShape weight_shape = BITensorShape(768, 6003);
    const std::string &weight_path =
            "/Users/mason/Desktop/Desktop/PythonProjects/dynamic_simple_ops/transformer_wte_weight.npy";
    BITensor weight = utils::create_type_tensor(
        weight_path, weight_shape,
        BIDataType::F16);
    // QATTest::print_tensor(weight, "weight");

    // 2. 输入一个input的权重
    BITensorShape input_shape = BITensorShape(3, 1);
    BITensor indices;
    indices.allocator()->init(BITensorInfo(input_shape, 1, BIDataType::U32));
    indices.allocator()->allocate();
    std::vector<uint32_t> indices_val{0, 2, 3};
    QATTest::fill_tensor_val_with_arr(indices, indices_val);
    QATTest::print_tensor(indices);

    // 3. 创建输出tensor
    BITensorShape o_shape = BITensorShape(768, 3, 1);
    BITensor o_tensor;
    o_tensor.allocator()->init(BITensorInfo(o_shape, 1, BIDataType::F16));
    o_tensor.allocator()->allocate();

    // 3. 创建NEGather
    BINEGather gather_op;
    gather_op.configure(&weight, &indices, &o_tensor, 1);
    gather_op.run();

    // std::cout << o_tensor. << std::endl;
    QATTest::print_tensor(o_tensor);
}

TEST(NEGatherTest, NEGatherDynamicTest) {
    // 1. 先读取numpy的Gather权重
    BITensorShape weight_shape = BITensorShape(768, 6003);
    const std::string &weight_path =
            "/Users/mason/Desktop/Desktop/PythonProjects/dynamic_simple_ops/transformer_wte_weight.npy";
    BITensor weight = utils::create_type_tensor(
        weight_path, weight_shape,
        BIDataType::F16);
    // QATTest::print_tensor(weight, "weight");

    // 2. 输入一个input的权重
    BITensorShape input_shape = BITensorShape(3, 1);
    BITensor indices;
    indices.allocator()->init(BITensorInfo(input_shape, 1, BIDataType::U32));
    indices.allocator()->allocate();
    std::vector<uint32_t> indices_val{0, 2, 3};
    QATTest::fill_tensor_val_with_arr(indices, indices_val);
    QATTest::print_tensor(indices);

    // 3. 创建输出tensor
    BITensorShape o_shape = BITensorShape(768, 3, 1);
    BITensor o_tensor;
    o_tensor.allocator()->init(BITensorInfo(o_shape, 1, BIDataType::F16));
    o_tensor.allocator()->allocate();

    // 3. 创建NEGather
    BINEGather gather_op;
    gather_op.configure(&weight, &indices, &o_tensor, 1);
    gather_op.run();

    // std::cout << o_tensor. << std::endl;
    QATTest::print_tensor(o_tensor);

    // 创建新的indices
    indices.allocator()->free();
    input_shape = BITensorShape(4, 1);
    indices.allocator()->init(BITensorInfo(input_shape, 1, BIDataType::U32));
    indices.allocator()->allocate();
    indices_val.emplace_back(4);
    QATTest::fill_tensor_val_with_arr(indices, indices_val);
    o_shape = BITensorShape(768, 4, 1);
    o_tensor.allocator()->init(BITensorInfo(o_shape, 1, BIDataType::F16));
    o_tensor.allocator()->allocate();

    gather_op.dynamic_configure(&indices, &o_tensor);
    gather_op.run();
    QATTest::print_tensor(o_tensor);
}

TEST(NERMSNormTest, RMSNormStaticTest) {
    // 1. 先确定Gamma数据
    BITensorShape gamma_shape = BITensorShape(768);
    const std::string &gamma_path = "/Users/mason/Desktop/Desktop/PythonProjects/dynamic_simple_ops/rms_norm1.npy";
    BITensor gamma = utils::create_type_tensor(gamma_path, gamma_shape, BIDataType::F16);
    // QATTest::print_tensor(gamma, "gamma");

    // 2. 确定输入张量信息
    BITensorShape input_shape = BITensorShape(768, 4);
    const std::string &input_path =
            "/Users/mason/Desktop/Desktop/PythonProjects/dynamic_simple_ops/rms_norm1_input.npy";
    BITensor input = utils::create_type_tensor(input_path, input_shape, BIDataType::F16);
    // QATTest::print_tensor(input, "input");

    // 3 输出张量
    BITensorShape o_shape = BITensorShape(768, 4);
    BITensor o_tensor;
    o_tensor.allocator()->init(BITensorInfo(o_shape, 1, BIDataType::F16));
    o_tensor.allocator()->allocate();

    // 4. 配置RMSNorm
    BINERMSNormLayer rms_layer;
    rms_layer.configure(&input, &gamma, &o_tensor);
    rms_layer.run();

    QATTest::print_tensor(o_tensor, "output");
}


TEST(NERMSNormTest, RMSNormDynamicTest) {
    int seq_len = 3;
    // 1. 先确定Gamma数据
    BITensorShape gamma_shape = BITensorShape(768);
    const std::string &gamma_path = "/Users/mason/Desktop/Desktop/PythonProjects/dynamic_simple_ops/rms_norm1.npy";
    BITensor gamma = utils::create_type_tensor(gamma_path, gamma_shape, BIDataType::F16);
    // QATTest::print_tensor(gamma, "gamma");

    // 2. 确定输入张量信息
    BITensorShape input_shape = BITensorShape(768, seq_len);
    std::string input_path =
            "/Users/mason/Desktop/Desktop/PythonProjects/dynamic_simple_ops/rms_norm1_input_sub.npy";
    BITensor input = utils::create_type_tensor(input_path, input_shape, BIDataType::F16);
    // QATTest::print_tensor(input, "input");

    // 3 输出张量
    BITensorShape o_shape = BITensorShape(768, seq_len);
    BITensor o_tensor;
    o_tensor.allocator()->init(BITensorInfo(o_shape, 1, BIDataType::F16));
    o_tensor.allocator()->allocate();

    // 4. 配置RMSNorm
    BINERMSNormLayer rms_layer;
    rms_layer.configure(&input, &gamma, &o_tensor);
    rms_layer.run();

    QATTest::print_tensor(o_tensor, "output");

    seq_len = 4;
    input.allocator()->free();
    input_shape = BITensorShape(768, seq_len);
    input_path =
            "/Users/mason/Desktop/Desktop/PythonProjects/dynamic_simple_ops/rms_norm1_input.npy";
    input = utils::create_type_tensor(input_path, input_shape, BIDataType::F16);

    o_tensor.allocator()->free();
    o_shape = BITensorShape(768, seq_len);
    o_tensor.allocator()->init(BITensorInfo(o_shape, 1, BIDataType::F16));
    o_tensor.allocator()->allocate();

    rms_layer.dynamic_configure(&input);
    rms_layer.run();
    QATTest::print_tensor(o_tensor, "output");
}

TEST(NESplitTest, NESplitStaticTest) {
    // 输入张量
    BITensor input, output_1, output_2, output_3;
    BITensorShape input_shape = BITensorShape(2304, 16, 20), output_shape = BITensorShape(768, 16, 20);
    input.allocator()->init(BITensorInfo(input_shape, 1, BIDataType::F16));
    input.allocator()->allocate();
    output_1.allocator()->init(BITensorInfo(output_shape, 1, BIDataType::F16));
    output_1.allocator()->allocate();
    output_2.allocator()->init(BITensorInfo(output_shape, 1, BIDataType::F16));
    output_2.allocator()->allocate();
    output_3.allocator()->init(BITensorInfo(output_shape, 1, BIDataType::F16));
    output_3.allocator()->allocate();
    std::vector<BIITensor *> outputs = {
        &output_1, &output_2, &output_3
    };

    // 配置参数
    BINESplit split_layer;
    split_layer.configure(&input, outputs, 0);

    split_layer.run();
    input_shape = BITensorShape(2304, 17, 30), output_shape = BITensorShape(768, 17, 30);
    input.allocator()->init(
        BITensorInfo(input_shape, 1, BIDataType::F16));
    input.allocator()->allocate();
    output_1.allocator()->init(BITensorInfo(output_shape, 1, BIDataType::F16));
    output_1.allocator()->allocate();
    output_2.allocator()->init(BITensorInfo(output_shape, 1, BIDataType::F16));
    output_2.allocator()->allocate();
    output_3.allocator()->init(BITensorInfo(output_shape, 1, BIDataType::F16));
    output_3.allocator()->allocate();
    QATTest::fill_new_tensor_val(input, static_cast<float16_t>(1));
    // split_layer.dynamic_configure(&input);
    // split_layer.run();
    // QATTest::print_tensor(output_3, "output");
}

TEST(NEReshapeLayer, NEReshapeStaticTest) {
    BITensor input, output;
    BITensorShape input_shape = BITensorShape(768, 1, 1), output_shape =
            BITensorShape(64, 12, 1, 1);
    input.allocator()->init(BITensorInfo(input_shape, 1, BIDataType::F32));
    input.allocator()->allocate();
    output.allocator()->init(BITensorInfo(output_shape, 1, BIDataType::F32));
    output.allocator()->allocate();
    QATTest::fill_from_one<float>(input);

    BINEReshapeLayer _reshape_f;
    _reshape_f.configure(&input, &output);
    _reshape_f.run();
    QATTest::print_tensor(output, "output");

    input_shape = BITensorShape(768, 20, 16), output_shape =
            BITensorShape(64, 12, 20, 16);
    input.allocator()->init(BITensorInfo(input_shape, 1, BIDataType::F32));
    input.allocator()->allocate();
    output.allocator()->init(BITensorInfo(output_shape, 1, BIDataType::F32));
    output.allocator()->allocate();
    QATTest::fill_from_one<float>(input);
    _reshape_f.dynamic_configure(&output);
    _reshape_f.run();
    QATTest::print_tensor(output, "output");
}

TEST(NETransposeLayer, TransposeStaticTest) {
    BITensor input, output;
    BITensorShape input_shape = BITensorShape(64, 12, 1, 1), output_shape =
            BITensorShape(64, 1, 12, 1);
    input.allocator()->init(BITensorInfo(input_shape, 1, BIDataType::F32));
    input.allocator()->allocate();
    output.allocator()->init(BITensorInfo(output_shape, 1, BIDataType::F32));
    output.allocator()->allocate();
    QATTest::fill_from_one<float>(input);

    BINEPermute permute_f;
    permute_f.configure(&input, &output, PermutationVector{0, 2, 1, 3});
    permute_f.run();
    QATTest::print_tensor(output, "output");
    QATTest::print_tensor_shape(output);

    input_shape = BITensorShape(64, 12, 16, 20), output_shape =
            BITensorShape(64, 16, 12, 20);
    input.allocator()->init(BITensorInfo(input_shape, 1, BIDataType::F32));
    input.allocator()->allocate();
    output.allocator()->init(BITensorInfo(output_shape, 1, BIDataType::F32));
    output.allocator()->allocate();
    QATTest::fill_from_one<float>(input);
    permute_f.dynamic_configure(&input, &output);
    permute_f.run();
    QATTest::print_tensor(output, "output");
    QATTest::print_tensor_shape(output);
}

TEST(NETransposeLayer, TransposeStaticTest2) {
    BITensor input, output;
    BITensorShape input_shape = BITensorShape(64, 12, 1, 1), output_shape =
            BITensorShape(1, 64, 12, 1);
    input.allocator()->init(BITensorInfo(input_shape, 1, BIDataType::F32));
    input.allocator()->allocate();
    output.allocator()->init(BITensorInfo(output_shape, 1, BIDataType::F32));
    output.allocator()->allocate();
    QATTest::fill_from_one<float>(input);

    BINEPermute permute_f;
    permute_f.configure(&input, &output, PermutationVector{2, 0, 1, 3});
    permute_f.run();
    // QATTest::print_tensor(output, "output");
    QATTest::print_tensor_shape(output);

    input_shape = BITensorShape(64, 12, 16, 20), output_shape =
            BITensorShape(16, 64, 12, 20);
    input.allocator()->init(BITensorInfo(input_shape, 1, BIDataType::F32));
    input.allocator()->allocate();
    output.allocator()->init(BITensorInfo(output_shape, 1, BIDataType::F32));
    output.allocator()->allocate();
    QATTest::fill_from_one<float>(input);
    permute_f.dynamic_configure(&input, &output);
    permute_f.run();
    // QATTest::print_tensor(output, "output");
    QATTest::print_tensor_shape(output);
}

TEST(NEAddLayer, AddLayerOps) {
    BITensor input, output;
    BITensorShape input_shape = BITensorShape(16, 16, 2, 1);
    input.allocator()->init(BITensorInfo(input_shape, 1, BIDataType::F32));
    input.allocator()->allocate();
    output.allocator()->init(BITensorInfo(input_shape, 1, BIDataType::F32));
    output.allocator()->allocate();
    QATTest::fill_from_one<float>(input);
    // 1. 创建条件掩码张量(U8类型)
    BITensorInfo mask_info(input_shape, 1, BIDataType::U8);
    BITensor mask;
    mask.allocator()->init(mask_info);
    mask.allocator()->allocate();
    // 2. 生成对角线掩码 - 对角线位置为1,其他为0
    BIWindow window;
    window.use_tensor_dimensions(mask.info()->tensor_shape());
    BIIterator mask_it(&mask, window);
    execute_window_loop(window, [&](const BICoordinates &id) {
                            auto x = id[0];
                            auto y = id[1];
                            *reinterpret_cast<uint8_t *>(mask_it.ptr()) = (x > y) ? 1 : 0;
                        },
                        mask_it);

    // 3. 创建全0张量作为y输入
    BITensor zeros;
    zeros.allocator()->init(*input.info());
    zeros.allocator()->allocate();
    std::fill_n(zeros.buffer(), zeros.info()->total_size(), 0);

    // 4. 配置NESelect算子
    BINESelect select;
    select.configure(&mask, &input, &zeros, &output);
    // 5. 运行算子
    select.run();
    // 清理临时张量
    mask.allocator()->free();
    zeros.allocator()->free();
    QATTest::print_tensor(output, "output");
}

TEST(DynamicTensor, DynamicTensorTest) {
    BITensor input, output, weight;
    BITensorShape input_shape = BITensorShape(4, 4, 1, 1);
    input.allocator()->init(BITensorInfo(input_shape, 1, BIDataType::F16));
    input.allocator()->allocate();
    output.allocator()->init(BITensorInfo(input_shape, 1, BIDataType::F16));
    output.allocator()->allocate();
    QATTest::fill_from_one<float>(input);
    BITensorShape weight_shape = BITensorShape(16, 16);
    weight.allocator()->init(BITensorInfo(weight_shape, 1, BIDataType::F16));
    weight.allocator()->allocate();
    BIWindow window;
    window.use_tensor_dimensions(weight.info()->tensor_shape());
    BIIterator mask_it(&weight, window);
    execute_window_loop(window, [&](const BICoordinates &id) {
                            auto x = id[0];
                            auto y = id[1];
                            *reinterpret_cast<float16_t *>(mask_it.ptr()) = (x > y)
                                                                                ? 0
                                                                                : -std::numeric_limits<
                                                                                    float>::infinity();
                        },
                        mask_it);
    BITensorShape sub_shape(4, 4); // 要提取 64x64 的子区域
    BITensorInfo sub_info(sub_shape, 1, BIDataType::F16);
    sub_info.set_format(Format::F16);

    // 3. 定义子张量的起始坐标
    BICoordinates coords(0, 0); // 从(32,32)位置开始提取

    // 4. 创建子张量
    BITensor sub_tensor;
    sub_tensor.allocator()->init(*weight.allocator(), coords, sub_info);
    QATTest::print_tensor(weight, "weight_tensor");
    QATTest::print_tensor(sub_tensor, "sub_tensor");

    BINEArithmeticAddition add_f;
    add_f.configure(&input, &sub_tensor, &output, BIConvertPolicy::SATURATE);
    add_f.run();
    QATTest::print_tensor(output, "output"); // 2. 定义要提取的子张量信息

    input_shape = BITensorShape(16, 16, 19, 10);
    input.allocator()->init(BITensorInfo(input_shape, 1, BIDataType::F16));
    input.allocator()->allocate();
    output.allocator()->init(BITensorInfo(input_shape, 1, BIDataType::F16));
    output.allocator()->allocate();
    QATTest::fill_from_one<float>(input);
    sub_shape = BITensorShape(16, 16);
    sub_info = BITensorInfo(sub_shape, 1, BIDataType::F16);
    sub_info.set_format(Format::F16);
    sub_tensor.allocator()->init(*weight.allocator(), coords, sub_info);
    add_f.dynamic_configure(&input, &sub_tensor, true);
    add_f.run();
    QATTest::print_tensor(output, "output2"); // 2. 定义要提取的子张量信息
}

TEST(DynamicTensor, DynamicSoftmax) {
    BITensor input, output;
    auto shape1 = BITensorShape(2, 4);
    input.allocator()->init(BITensorInfo(shape1, 1, BIDataType::F16));
    input.allocator()->allocate();
    QATTest::fill_new_tensor_val(input, 1);
    output.allocator()->init(BITensorInfo(shape1, 1, BIDataType::F16));
    output.allocator()->allocate();

    BINESoftmaxLayerGeneric<false> softmax_f;
    softmax_f.configure(&input, &output, 1.0f, 0);
    softmax_f.run();
    QATTest::print_tensor(output, "output1");

    auto shape2 = BITensorShape(4, 2);
    input.allocator()->init(BITensorInfo(shape2, 1, BIDataType::F16));
    input.allocator()->allocate();
    QATTest::fill_new_tensor_val(input, 1);
    output.allocator()->init(BITensorInfo(shape2, 1, BIDataType::F16));
    output.allocator()->allocate();

    softmax_f.dynamic_configure();
    softmax_f.run();
    QATTest::print_tensor(output, "output2");
}

TEST(DynamicTensor, DynamicGeLU) {
    BITensorShape input_shape = BITensorShape(3072, 3, 1);
    std::string input_path = "/Users/mason/Desktop/Desktop/PythonProjects/dynamic_simple_ops/q_act_before.npy";
    BIQuantizationInfo qin_info = BIQuantizationInfo(0.2122f, -9);
    BITensor input = utils::create_type_tensor(input_path, input_shape, BIDataType::QASYMM8_SIGNED);
    input.info()->set_quantization_info(qin_info);
    QATTest::print_new_tensor(input);
    BIQuantizationInfo qout_info = BIQuantizationInfo(0.1137f, -127);
    BITensor output;
    output.allocator()->init(BITensorInfo(input_shape, 1, BIDataType::QASYMM8_SIGNED, qout_info));
    output.allocator()->allocate();

    BINEActivationLayer activation_layer;
    BIActivationLayerInfo activation_layer_info(BIActivationFunction::GELU);
    activation_layer.configure(&input, &output, activation_layer_info);
    activation_layer.run();

    QATTest::print_tensor(output, "output");
    BINEDequantizationLayer dequantization_layer;

    // 动态修改input
    input_shape = BITensorShape(3072, 5, 1);
    input_path = "/Users/mason/Desktop/Desktop/PythonProjects/dynamic_simple_ops/q_act_before_1.npy";
    qin_info = BIQuantizationInfo(0.2122f, -9);
    input = utils::create_type_tensor(input_path, input_shape, BIDataType::QASYMM8_SIGNED);
    input.info()->set_quantization_info(qin_info);

    qout_info = BIQuantizationInfo(0.1137f, -127);
    output.allocator()->init(BITensorInfo(input_shape, 1, BIDataType::QASYMM8_SIGNED, qout_info));
    output.allocator()->allocate();
    activation_layer.dynamic_configure(&input);
    activation_layer.run();
    QATTest::print_tensor(output, "output2");
    // BITensor deq_output;
    // deq_output.allocator()->init(BITensorInfo(input_shape, 1, BIDataType::F16));
    // deq_output.allocator()->allocate();
    // dequantization_layer.configure(&output, &deq_output);
    // dequantization_layer.run();
    //
    // QATTest::print_tensor(deq_output, "output2");
}

TEST(DynamicTensor, NOquantDynamicGeLU) {
    BITensorShape input_shape = BITensorShape(3072, 3, 1);
    const std::string &input_path = "/Users/mason/Desktop/Desktop/PythonProjects/dynamic_simple_ops/act_before.npy";
    BITensor input = utils::create_type_tensor(input_path, input_shape, BIDataType::F16);
    QATTest::print_new_tensor(input);
    BITensor output;
    output.allocator()->init(BITensorInfo(input_shape, 1, BIDataType::F16));
    output.allocator()->allocate();

    BINEActivationLayer activation_layer;
    BIActivationLayerInfo activation_layer_info(BIActivationFunction::GELU);
    activation_layer.configure(&input, &output, activation_layer_info);
    activation_layer.run();

    QATTest::print_tensor(output, "output");
}

TEST(INT8GPT_2, INT8GPT2Dynamic) {
    const int batch_size = 1;
    const int seq_len = 4;
    // 1. 先初始化输入矩阵
    const std::string &input_path = "/Users/mason/Desktop/Desktop/PythonProjects/quantize_gpt_qat/mlp_input.npy";
    BITensor input = QATTest::create_norm_input(std::vector<int>{1, 4, 768}, input_path);
    QATTest::print_tensor(input, "input");
    // 2. 初始化gamma张量
    const std::string &gamma_path = "/Users/mason/Desktop/Desktop/PythonProjects/quantize_gpt_qat/mlp_rms_gamma.npy";
    BITensor gamma = QATTest::create_norm_input(std::vector<int>{768}, gamma_path);
    // 3. 初始化fc_weights的权重
    const std::string &c_fc_weights_path =
            "/Users/mason/Desktop/Desktop/PythonProjects/quantize_gpt_qat/reordered_c_fc_weights.npy";
    std::vector<float> c_fc_weights_scales;
    // 量化信息
    BIQuantizationInfo c_fc_weight_qinfo;
    std::ifstream c_fc_weights_scale_file(
        "/Users/mason/Desktop/Desktop/PythonProjects/quantize_gpt_qat/c_fc_scales.txt");
    float value;

    while (c_fc_weights_scale_file >> value) {
        c_fc_weights_scales.push_back(value);
    }
    BITensor c_fc_weights = QATTest::create_per_channel(c_fc_weights_scales, std::vector{768, 3072},
                                                        c_fc_weight_qinfo, c_fc_weights_path);
    // 4. 初始化fc_bias
    const std::string &c_fc_bias_path = "/Users/mason/Desktop/Desktop/PythonProjects/quantize_gpt_qat/c_fc_bias.npy";
    BITensor c_fc_bias = QATTest::create_norm_bias(3072, c_fc_bias_path);
    // QATTest::print_tensor(c_fc_bias, "c_fc_bias");
    // 5. 输出张量
    BITensor output;
    output.allocator()->init(BITensorInfo(BITensorShape(768, seq_len, batch_size), 1, BIDataType::F16));
    output.allocator()->allocate();

    // 6. proj的权重
    const std::string &c_proj_path = "/Users/mason/Desktop/Desktop/PythonProjects/quantize_gpt_qat/c_proj_weights.npy";
    BITensor c_proj_weight = QATTest::create_norm_input(std::vector<int>{3072, 768}, c_proj_path);
    // QATTest::print_tensor(c_proj_weight, "c_proj");

    const std::string &c_proj_bias_path =
            "/Users/mason/Desktop/Desktop/PythonProjects/quantize_gpt_qat/c_proj_bias.npy";
    BITensor c_proj_bias = QATTest::create_norm_input(std::vector<int>{768}, c_proj_bias_path);
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
    _mlp_layer.run();

    // 1. 先用输出结果进行相加
    BITensorShape add_output_shape(input.info()->tensor_shape());
    BITensor add_output;
    add_output.allocator()->init(BITensorInfo(add_output_shape, 1, BIDataType::F16));
    add_output.allocator()->allocate();
    BINEArithmeticAddition add_f;
    add_f.configure(&output, &input, &add_output, BIConvertPolicy::SATURATE);
    add_f.run();

    // 2. 对结果再进行一次归一化
    BITensor mlp_after_gamma = QATTest::create_norm_input(std::vector{768},
                                                          "/Users/mason/Desktop/Desktop/PythonProjects/quantize_gpt_qat/mlp_after_rms_gamma.npy");

    BITensor mlp_rms_output;
    mlp_rms_output.allocator()->init(BITensorInfo(input.info()->tensor_shape(), 1, BIDataType::F16));
    mlp_rms_output.allocator()->allocate();

    BINERMSNormLayer rms_norm_layer;
    rms_norm_layer.configure(&add_output, &mlp_after_gamma, &mlp_rms_output);
    rms_norm_layer.run();

    // 3. 对输出结果进行LMHead操作
    BITensor lm_head_weights = QATTest::create_norm_input(std::vector{768, 6003},
                                                          "/Users/mason/Desktop/Desktop/PythonProjects/quantize_gpt_qat/lm_head_weights.npy");

    BITensor lm_head_output;
    lm_head_output.allocator()->init(BITensorInfo(BITensorShape(6003, seq_len, batch_size), 1, BIDataType::F16));
    lm_head_output.allocator()->allocate();

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
    lm_head_layer.configure(&mlp_rms_output, &lm_head_weights, nullptr, &lm_head_output, 1.0f, 1.0f, gemm_info);
    lm_head_layer.run();
    QATTest::print_tensor(lm_head_output);

    BITensor ids;
    ids.allocator()->init(BITensorInfo(BITensorShape(seq_len, batch_size), 1, BIDataType::S32));
    ids.allocator()->allocate();

    BINEArgMinMaxLayer arg_minmax_layer;
    arg_minmax_layer.configure(&lm_head_output, 0, &ids, BIReductionOperation::ARG_IDX_MAX);
    arg_minmax_layer.run();

    QATTest::print_tensor(lm_head_output, "lm_head_output");


    QATTest::print_tensor(ids, "ids");
}


