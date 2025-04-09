//
// Created by Mason on 2025/3/27.
//

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <runtime/neon/bi_ne_functions.h>

#include "data/core/utils/quantization/asymm_helpers.hpp"
#include "utils/utils.hpp"

namespace GemmTest {
    using namespace BatmanInfer;

    void create_tensor_by(
        const float scale,
        const int offset,
        const std::vector<int> &tensor_shape,
        const std::vector<int8_t> &tensor_data,
        BITensor *tensor) {
        // 1. Create Quantization info
        BIQuantizationInfo qinfo = BIQuantizationInfo(scale, offset);
        tensor->allocator()->init(
            BITensorInfo(
                BITensorShape(tensor_shape[1], tensor_shape[0]),
                1,
                BIDataType::QASYMM8_SIGNED,
                qinfo));
        tensor->allocator()->allocate();
        std::memcpy(tensor->buffer(), tensor_data.data(), tensor->info()->total_size());
    }

    void create_bias_tensor_by(const int &dim_shape,
                               const std::vector<int32_t> &tensor_data,
                               BITensor *tensor) {
        tensor->allocator()->init(
            BITensorInfo(
                BITensorShape(dim_shape),
                1,
                BIDataType::S32));
        tensor->allocator()->allocate();
        std::memcpy(tensor->buffer(), tensor_data.data(), tensor->info()->total_size());
    }

    void create_per_channel_tensor(
        const std::vector<float> &weight_scales,
        const std::vector<int8_t> &weights_data,
        const std::vector<int> &tensor_shape,
        BITensor *tensor) {
        BIQuantizationInfo qinfo = BIQuantizationInfo(weight_scales);
        tensor->allocator()->init(
            BITensorInfo(
                BITensorShape(tensor_shape[1], tensor_shape[0]),
                1,
                BIDataType::QSYMM8_PER_CHANNEL,
                qinfo));
        tensor->allocator()->allocate();
        std::memcpy(tensor->buffer(), weights_data.data(), tensor->info()->total_size());
    }

    void print_tensor(const BITensor &tensor, const std::string &name = "exmaple") {
        BIIOFormatInfo format;
        format.element_delim = ", "; // 元素之间用逗号分隔
        format.row_delim = "\n"; // 每行换行
        format.align_columns = 1; // 对齐列

        tensor.print(std::cout, format);
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

    BITensor create_npy_bias(const int &dim_size,
                             const std::string &file_path = "") {
        auto bias_shape = BITensorShape(dim_size);
        BITensor bias = utils::create_type_tensor(file_path, bias_shape, BIDataType::S32);
        return bias;
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
}

TEST(DYNAMIC_GEMMLOWP, GRAPH_GEMMLOWP) {
    using namespace BatmanInfer;
    // 1. 初始化输入算子
    std::vector<int8_t> input_data{-1, 68};
    std::vector input_shape{1, 2};
    BITensor input;
    GemmTest::create_tensor_by(0.011764705882352941, 43, input_shape, input_data, &input);
    std::vector<int8_t> weights_data{127, 127, -127, -45, -20, 88, 81, 127};
    std::vector weights_shape{2, 4};
    BITensor weight;
    GemmTest::create_per_channel_tensor(std::vector<float>{0.0038, 0.0103, 0.0058, 0.0133}, weights_data, weights_shape,
                                        &weight);
    GemmTest::print_tensor(input, "input");
    GemmTest::print_tensor(weight, "weights");

    BITensor bias;
    GemmTest::create_bias_tensor_by(4, std::vector<int32_t>{10795, 10794, -10795, -3859}, &bias);


    // 2. 输出算子
    BITensor output;
    output.allocator()->init(BITensorInfo(BITensorShape(4, 1), 1, BIDataType::QASYMM8_SIGNED, BIQuantizationInfo(
                                              0.014115710585725074,
                                              -94)));
    output.allocator()->allocate();
    BIGEMMLowpOutputStageInfo info;
    info.type = BIGEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT;
    info.gemmlowp_offset = output.info()->quantization_info().uniform().offset;
    info.gemmlowp_min_bound = -128;
    info.gemmlowp_max_bound = 127;
    info.output_data_type = BIDataType::QASYMM8_SIGNED;
    quantization::calculate_quantized_multipliers(input.info()->quantization_info(),
                                                  weight.info()->quantization_info(),
                                                  output.info()->quantization_info(), info);
    GEMMInfo gemm_info = GEMMInfo(false, false, true, false, false, false, info, false, false, false,
                                  BIActivationLayerInfo(), false, BIWeightFormat::UNSPECIFIED, false);
    BINEGEMMLowpMatrixMultipleCore gemmlowp_mm_score;
    gemmlowp_mm_score.configure(&input, &weight, &bias, &output, gemm_info);
    // BatmanInfer::invert_qinfo_offset(input);
    // BatmanInfer::invert_qinfo_offset(weight);
    // gemmlowp_mm_score.update_quantization_parameters();
    gemmlowp_mm_score.run();
    GemmTest::print_tensor(output, "output");
    BITensor final_output;
    final_output.allocator()->init(BITensorInfo(BITensorShape(4, 1), 1, BIDataType::F16));
    final_output.allocator()->allocate();

    // 6. 反量化
    BINEDequantizationLayer dequantization_layer;
    dequantization_layer.configure(&output, &final_output);
    dequantization_layer.run();

    GemmTest::print_tensor(final_output, "final output");
}

TEST(DYNAMIC_GEMMLOWP_ERROR, GRAPH_GEMMLOWP_ERROR) {
    using namespace BatmanInfer;
    // 1. 初始化输入算子
    std::vector input_shape{1, 4, 768};
    BIQuantizationInfo q_input_info;
    const std::string &input_path =
            "/Users/mason/Desktop/Desktop/PythonProjects/gemmlowp_compare/confrim_input_tensor.npy";
    BITensor input = GemmTest::create_qasymm8(0.011764705882352941, 43, input_shape, q_input_info, input_path);
    GemmTest::print_tensor(input, "input");

    BITensor origin_in;
    origin_in.allocator()->init(BITensorInfo(BITensorShape(768, 4, 1), 1, BIDataType::F16));
    origin_in.allocator()->allocate();
    BINEDequantizationLayer dequantization_layer;
    input.info()->set_quantization_info(BIQuantizationInfo(0.011764705882352941, -43));
    dequantization_layer.configure(&input, &origin_in);
    dequantization_layer.run();
    // GemmTest::print_tensor(origin_in, "origin");

    // 2. 获取per channel量化的weights
    std::vector<float> weight_scales;
    std::ifstream in_file("/Users/mason/Desktop/Desktop/PythonProjects/gemmlowp_compare/confrim_weight_scale.txt");
    float value;

    while (in_file >> value) {
        weight_scales.push_back(value);
    }
    std::vector weight_shape{768, 3072};
    BIQuantizationInfo q_weight_info;
    const std::string &weight_path =
            "/Users/mason/Desktop/Desktop/PythonProjects/gemmlowp_compare/confrim_weight_tensor.npy";
    BITensor weight = GemmTest::create_per_channel(weight_scales, weight_shape, q_weight_info, weight_path);
    BITensor origin_weight;
    origin_weight.allocator()->init(BITensorInfo(BITensorShape(3072, 768), 1, BIDataType::F16));
    origin_weight.allocator()->allocate();
    dequantization_layer.configure(&weight, &origin_weight);
    dequantization_layer.run();

    // 3. 获取bias矩阵
    const std::string &bias_path =
            "/Users/mason/Desktop/Desktop/PythonProjects/gemmlowp_compare/confrim_bias_tensor.npy";
    BITensor bias = GemmTest::create_npy_bias(3072, bias_path);


    // GemmTest::print_tensor(origin_weight, "weight");
    BITensor s32_output;
    s32_output.allocator()->init(BITensorInfo(BITensorShape(3072, 4, 1), 1, BIDataType::S32));
    s32_output.allocator()->allocate();

    BINEGEMMLowpMatrixMultipleCore gemmlowp_mm_score;
    input.info()->set_quantization_info(BIQuantizationInfo(0.011764705882352941, 43));
    gemmlowp_mm_score.configure(&input, &weight, nullptr, &s32_output);
    gemmlowp_mm_score.run();
    GemmTest::print_tensor(s32_output, "output32");
    // 输出矩阵
    BITensor output;
    output.allocator()->init(BITensorInfo(BITensorShape(3072, 4, 1), 1, BIDataType::QASYMM8_SIGNED, BIQuantizationInfo(
                                              6.0953759885301775,
                                              132)));
    output.allocator()->allocate();
    BIGEMMLowpOutputStageInfo info;
    // 执行运行
    info.type = BIGEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT;
    info.gemmlowp_offset = output.info()->quantization_info().uniform().offset;
    info.gemmlowp_min_bound = -128;
    info.gemmlowp_max_bound = 127;
    info.output_data_type = BIDataType::QASYMM8_SIGNED;
    // input.info()->set_quantization_info(BIQuantizationInfo(0.011764705882352941, -43));
    quantization::calculate_quantized_multipliers(input.info()->quantization_info(),
                                                  weight.info()->quantization_info(),
                                                  output.info()->quantization_info(), info);
    GEMMInfo gemm_info = GEMMInfo(false, false, true, false, false, false, info, false, false, false,
                                  BIActivationLayerInfo(), false, BIWeightFormat::UNSPECIFIED, false);

    BINEGEMMLowpOutputStage n_binegemm_lowp_output_stage;
    n_binegemm_lowp_output_stage.configure(&s32_output, &bias, &output, info);
    n_binegemm_lowp_output_stage.run();
    GemmTest::print_tensor(output, "output");

    BITensor final_output;
    final_output.allocator()->init(BITensorInfo(BITensorShape(3072, 4, 1), 1, BIDataType::F16));
    final_output.allocator()->allocate();
    dequantization_layer.configure(&output, &final_output);
    dequantization_layer.run();
    GemmTest::print_tensor(final_output, "final output");
}
