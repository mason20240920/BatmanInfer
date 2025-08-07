//
// Created by Mason on 2025/8/1.
//
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <runtime/neon/bi_ne_functions.h>
#include <runtime/bi_tensor.hpp>
#include <utils/utils.hpp>
#include "function_info/bi_MatMulInfo.h"
#include "runtime/bi_scheduler.hpp"
#include <thread>

#include "runtime/neon/bi_ne_scheduler.hpp"
#include "runtime/neon/functions/BINEIntentMGPTBlock.hpp"
#include "runtime/omp/bi_imp_scheduler.hpp"

namespace OperatorTest {
    using namespace BatmanInfer;

    /**
     * @brief 打印数据的位移
     * @param data_ptr
     * @param move_size
     */
    template<typename T>
    void print_offset(void *data_ptr, const size_t move_size) {
        T *p = static_cast<T *>(data_ptr);
        for (size_t i = 0; i < move_size; i++) {
            std::cout << p[i] << ", ";
        }
        std::cout << std::endl;
    }

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

    void concat_tensor(const BITensor &tensor,
                       std::vector<int> &output_ids) {
        // 元素开始起点
        const auto start_offset = static_cast<int>(tensor.info()->offset_first_element_in_bytes());
        const uint8_t *ptr = tensor.buffer() + start_offset;
        const auto input_s32 = static_cast<int32_t>(*ptr);
        output_ids.push_back(input_s32);
    }

    template<typename T>
    void print_output_info(const std::vector<T> &output_ids) {
        for (const auto &id: output_ids) {
            std::cout << id << ", ";
        }
        std::cout << std::endl;
    }

    void print_score(const std::vector<float> &scores, const size_t dim_size) {
        for (int i = 0; i < scores.size(); i++) {
            std::cout << scores[i] << "\t";
            if ((i + 1) % dim_size == 0) {
                std::cout << std::endl;
            }
        }
        std::cout << std::endl;
    }

    /**
     * @brief 获取索引的最大值
     * @param input_tensor: 张量信息
     * @param tensor_ids: 张量id信息
     * @param ret: 返回值id
     * @return
     */
    void get_index_val(const BITensor &input_tensor, const std::vector<int> &tensor_ids, std::vector<float> &ret) {
        ret.clear();
        const auto dim_size = input_tensor.info()->tensor_shape()[0];
        auto *data_ptr = reinterpret_cast<float16_t *>(input_tensor.buffer());
        const size_t ret_size = tensor_ids.size();
        for (size_t i = 0; i < ret_size; i++) {
            ret.push_back(static_cast<float>(data_ptr[tensor_ids[i] + i * dim_size]));
        }
    }

    void get_s32_val(const BITensor &input_tensor, std::vector<int> &ret) {
        ret.clear();
        const auto *data_ptr = reinterpret_cast<int32_t *>(input_tensor.buffer());
        const size_t ret_size = input_tensor.info()->tensor_shape().total_size();
        for (size_t i = 0; i < ret_size; i++) {
            ret.push_back(static_cast<int>(data_ptr[i]));
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

    template<typename T>
    void fill_tensor_val_with_arr(const BatmanInfer::BITensor &tensor, const std::vector<T> val) {
        auto tensor_ptr = reinterpret_cast<T *>(tensor.buffer());
        const size_t num_elements = tensor.info()->tensor_shape().total_size(); // 获取元素数量
        for (size_t i = 0; i < num_elements; i++) {
            tensor_ptr[i] = val[i];
        }
    }

    void fill_tensor_with_repeat_arr(std::vector<uint32_t> &input_ids, const int repeat_times,
                                     std::vector<uint32_t> input_id) {
        input_ids.clear();
        for (int i = 0; i < repeat_times; i++) {
            for (uint32_t j: input_id) {
                input_ids.emplace_back(j);
            }
        }
    }


    template<typename T>
    void fill_tensor_val_with_index(const BITensor &tensor) {
        auto tensor_ptr = reinterpret_cast<
            T *>(tensor.buffer());
        const size_t num_elements = tensor.info()->tensor_shape().total_size();
        // 获取元素数量
        for (size_t i = 0; i < num_elements; i++) {
            tensor_ptr[i] = static_cast<T>(i);
        }
    }

    template<typename T>
    void fill_tensor_val_with_index_2(const BITensor &tensor) {
        auto tensor_ptr = reinterpret_cast<
            T *>(tensor.buffer());
        const size_t num_elements = tensor.info()->tensor_shape().total_size();
        // 获取元素数量
        for (size_t i = 0; i < num_elements; i++) {
            tensor_ptr[i] = static_cast<T>(i + 2);
        }
    }

    // --- Function Definition ---

    std::optional<int> readIntegerFromFile(const std::string &filename, int line_number) {
        std::ifstream inputFile(filename);

        if (!inputFile.is_open()) {
            std::cerr << "Error: Could not open file '" << filename << "'" << std::endl;
            return std::nullopt; // Return empty optional
        }

        std::string line;
        int current_line = 0;

        // Loop to the desired line
        while (std::getline(inputFile, line)) {
            current_line++;
            if (current_line == line_number) {
                try {
                    // Attempt to convert the line to an integer
                    return std::stoi(line); // Success! Return the value.
                } catch (const std::invalid_argument &) {
                    std::cerr << "Error: Line " << line_number << " does not contain a valid number." << std::endl;
                    return std::nullopt;
                } catch (const std::out_of_range &) {
                    std::cerr << "Error: Number on line " << line_number << " is out of range for an integer." <<
                            std::endl;
                    return std::nullopt;
                }
            }
        }

        // If we finished the loop and didn't find the line
        if (current_line < line_number) {
            std::cerr << "Error: File has only " << current_line << " lines, cannot read line " << line_number << "." <<
                    std::endl;
        }

        return std::nullopt; // Return empty if line not found or file is empty
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

    /**
     * @brief 进行层归一化的推理CPU
     * @param out
     * @param mean
     * @param rstd
     * @param inp
     * @param weights
     * @param bias
     * @param B
     * @param T
     * @param C
     */
    void layer_norm_forward_cpu(float *out,
                                float *mean,
                                float *rstd,
                                const float *inp,
                                const float *weights,
                                const float *bias,
                                int B,
                                int T,
                                int C) {
        float eps = 1e-5f;
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                // seek to the input position inp[b,t,:]
                // 输入到某个batch, 某个sequence index
                const float *x = inp + b * T * C + t * C;
                // 计算均值
                float m = 0.0f;
                for (int i = 0; i < C; i++)
                    m += x[i];
                m = m / C;
                // 计算方差(不需要任何偏置值)
                float v = 0.0f;
                for (int i = 0; i < C; i++) {
                    float xshift = x[i] - m;
                    v += xshift * xshift;
                }
                v = v / C;
                // 计算rstd
                float s = 1.0f / sqrtf(v + eps);
                // 找到输出的位置 [b, t, :]
                float *out_bt = out + b * T * C + t * C;
                for (int i = 0; i < C; i++) {
                    float n = (s * (x[i] - m)); // 归一化输出
                    float o = n * weights[i] + bias[i]; // scale and shift it
                    out_bt[i] = o; // write
                }
                // cache the mean and rstd for the backward pass later
                mean[b * T + t] = m;
                rstd[b * T + t] = s;
            }
        }
    }

    // 辅助函数，用于打印数组内容
    void print_array(const char *name, const float *arr, int size, int limit = 10) {
        std::cout << name << ": [";
        for (int i = 0; i < std::min(size, limit); ++i) {
            std::cout << arr[i] << (i == std::min(size, limit) - 1 ? "" : ", ");
        }
        if (size > limit) {
            std::cout << "...";
        }
        std::cout << "]" << std::endl;
    }
}

TEST(KVCaches, LayerNormOP) {
    // 1. 定义维度
    int B = 2;
    int T = 3;
    int C = 4;

    // 2. 分配内存
    float *inp = new float[B * T * C];
    float *weights = new float[C];
    float *bias = new float[C];
    float *out = new float[B * T * C];
    float *mean = new float[B * T];
    float *rstd = new float[B * T]; // 3. 初始化数据
    // 初始化输入数据 (随便填一些值)
    for (int i = 0; i < B * T * C; ++i) {
        inp[i] = static_cast<float>(i);
    }
    // 初始化权重为1，偏置为0 (标准的层归一化)
    for (int i = 0; i < C; ++i) {
        weights[i] = 1.0f;
        bias[i] = 0.0f;
    }
    // 我们可以给权重和偏置设置一些不同的值来观察效果
    weights[0] = 1.5f;
    bias[1] = 0.5f;


    std::cout << "--- 输入数据 ---" << std::endl;
    OperatorTest::print_array("Input (inp)", inp, B * T * C);
    OperatorTest::print_array("Weights (gamma)", weights, C);
    OperatorTest::print_array("Bias (beta)", bias, C);
    std::cout << std::endl;

    // 4. 调用函数
    std::cout << "--- 调用 layer_norm_forward_cpu ---" << std::endl;
    OperatorTest::layer_norm_forward_cpu(out, mean, rstd, inp, weights, bias, B, T, C);
    std::cout << "函数执行完毕！" << std::endl << std::endl;

    // 5. 查看结果
    std::cout << "--- 输出结果 ---" << std::endl;
    OperatorTest::print_array("Output (out)", out, B * T * C, 24);
    OperatorTest::print_array("Cached Mean", mean, B * T);
    OperatorTest::print_array("Cached Rstd", rstd, B * T);

    // 6. 释放内存
    delete[] inp;
    delete[] weights;
    delete[] bias;
    delete[] out;
    delete[] mean;
    delete[] rstd;
}

TEST(LayerNorm, NELayerNormOp) {
    using namespace BatmanInfer;
    BITensorShape gather_weight_shape(768);
    const std::string &weight_path = "./gpt2_res/gamma.npy";
    const std::string &beta_path = "./gpt2_res/beta.npy";
    BITensor weight = utils::create_type_tensor(weight_path,
                                                gather_weight_shape,
                                                BIDataType::F16);
    BITensor bias = utils::create_type_tensor(beta_path,
                                              gather_weight_shape,
                                              BIDataType::F16);

    const std::string &input_path = "./gpt2_res/input.npy";
    BITensorShape input_tensor_shape(768, 16, 20);
    BITensor input = utils::create_type_tensor(input_path, input_tensor_shape, BIDataType::F16);

    BITensor output;
    output.allocator()->init(BITensorInfo(input_tensor_shape, 1, BIDataType::F16));
    output.allocator()->allocate();

    BINELayerNormLayer layer_norm;
    layer_norm.configure(&input, &weight, &bias, &output);
    layer_norm.run();

    OperatorTest::print_tensor(output);
}

TEST(LayerNorm, TestMultiGPTBlock) {
    using namespace BatmanInfer;
    // 前置参数
    std::vector<int> output_ids{};
    std::vector<float> scores{};
    // 需要复制传入数组的值
    std::vector<uint32_t> input_ids{};
    // 1. 先初始化KV Cache的数据
    constexpr int num_head = 12;
    constexpr int head_dim = 64;
    constexpr int hidden_size = 768;
    constexpr int max_batch_size = 20;
    constexpr int max_seq = 64;
    constexpr int block_layer_num = 1;
    BIScheduler::get().set_num_threads(2);

    // 开始的
    int batch_size = 1;
    int seq_len = 12;
    // 1. 初始化一个最大input算子
    BITensor input_tensor;
    const BITensorShape input_tensor_shape(max_seq, max_batch_size);
    input_tensor.allocator()->init(BITensorInfo(input_tensor_shape, 1, BIDataType::U32));
    input_tensor.allocator()->allocate();

    // 1.1 初始化sub input
    BITensor sub_input_tensor;
    BITensorShape sub_input_tensor_shape(seq_len, batch_size);
    BITensorInfo input_info(sub_input_tensor_shape, 1, BIDataType::U32);
    input_info.set_format(Format::U32);
    sub_input_tensor.allocator()->init(*input_tensor.allocator(), input_info);
    std::vector<uint32_t> indices_data{101, 6821, 3221, 2523,  719,  722, 1184, 4638,  752, 2658,  749,  102};
    OperatorTest::fill_tensor_val_with_arr(sub_input_tensor, indices_data);

    // 2. 转换token的Gather, Gather权重
    BITensorShape gather_weight_shape(hidden_size, 21128);
    const std::string &weight_path = "./intent_res/transformer_wte_weight.npy";
    BITensor weight = utils::create_type_tensor(weight_path,
                                                gather_weight_shape,
                                                BIDataType::F16);

    // 2.1 Gather的输出
    BITensor gather_o_tensor, sub_gather_o_tensor;
    BITensorShape gather_o_tensor_shape(hidden_size, max_seq, max_batch_size);
    BITensorShape sub_gather_o_shape(hidden_size, seq_len, batch_size);

    gather_o_tensor.allocator()->init(BITensorInfo(gather_o_tensor_shape,
                                                   1,
                                                   BIDataType::F16));
    gather_o_tensor.allocator()->allocate();
    BITensorInfo sub_gather_info(sub_gather_o_shape, 1, BIDataType::F16);
    sub_gather_info.set_format(Format::F16);
    sub_gather_o_tensor.allocator()->init(*gather_o_tensor.allocator(), sub_gather_info);

    // 2.2 Gather运行
    BINEGather gather_layer;
    gather_layer.configure(&weight, &sub_input_tensor, &sub_gather_o_tensor, 1);
    gather_layer.run();

    // 3. 位置信息编码信息
    BITensorShape add_wte_weight_shape(hidden_size, max_seq);
    const std::string &add_wte_weight_path = "./intent_res/add_wte_weights.npy";
    BITensor add_wte_weight = utils::create_type_tensor(add_wte_weight_path,
                                                        add_wte_weight_shape,
                                                        BIDataType::F16);
    BITensor sub_add_weight;
    BITensorShape sub_add_weight_shape(768, seq_len);
    BITensorInfo sub_add_weight_info(sub_add_weight_shape, 1, BIDataType::F16);
    sub_add_weight_info.set_format(Format::F16);
    sub_add_weight.allocator()->init(*add_wte_weight.allocator(), sub_add_weight_info);

    // 3.1 位置编码输出矩阵
    BITensor wte_o_tensor, sub_wte_o_tensor;
    wte_o_tensor.allocator()->init(BITensorInfo(gather_o_tensor_shape, 1, BIDataType::F16));
    wte_o_tensor.allocator()->allocate();
    sub_wte_o_tensor.allocator()->init(*wte_o_tensor.allocator(), sub_gather_info);

    BINEArithmeticAddition wte_layer;
    wte_layer.configure(&sub_gather_o_tensor, &sub_add_weight, &sub_wte_o_tensor, BIConvertPolicy::SATURATE);
    wte_layer.run();

    // OperatorTest::print_tensor(sub_wte_o_tensor);


    // 4 分割输出结果
    BITensorShape gpt_o_tensor_shape(hidden_size, max_seq, max_batch_size);
    BITensorShape sub_gpt_o_tensor_shape(hidden_size, seq_len, batch_size);
    BITensorInfo sub_gpt_o_t_info = BITensorInfo(sub_gpt_o_tensor_shape, 1, BIDataType::F16);
    sub_gpt_o_t_info.set_format(Format::F16);

    // 5. GPT Block的结构输出
    BITensorShape gpt_ln_1_weight_shape(hidden_size);
    const std::string &gamma_weights_path = "./intent_res/ln_1_weight_1.npy";
    BITensor gpt_ln_1_weights = utils::create_type_tensor(gamma_weights_path,
                                                          gpt_ln_1_weight_shape,
                                                          BIDataType::F16);
    const std::string &ln_1_bias_path = "./intent_res/ln_1_bias_1.npy";
    BITensor gpt_ln_1_bias = utils::create_type_tensor(ln_1_bias_path,
                                                       gpt_ln_1_weight_shape,
                                                       BIDataType::F16);
    BITensorShape gpt_c_attn_weights_shape(hidden_size * 3, hidden_size);
    const std::string &gpt_c_attn_weights_path = "./intent_res/c_attn_weights_1.npy";
    BITensor gpt_c_attn_weights = utils::create_type_tensor(gpt_c_attn_weights_path,
                                                            gpt_c_attn_weights_shape,
                                                            BIDataType::F16);
    BITensorShape gpt_c_attn_bias_shape(hidden_size * 3);
    const std::string &gpt_c_attn_bias_path = "./intent_res/c_attn_bias_1.npy";
    BITensor gpt_c_attn_bias = utils::create_type_tensor(gpt_c_attn_bias_path,
                                                         gpt_c_attn_bias_shape,
                                                         BIDataType::F16);
    BITensorShape attn_c_proj_weights_shape(hidden_size, hidden_size);
    const std::string &attn_c_proj_weights_path = "./input_res/p_attn_weights.npy";
    BITensor attn_c_proj_weights = utils::create_type_tensor(attn_c_proj_weights_path, attn_c_proj_weights_shape,
                                                             BIDataType::F16);
    BITensorShape attn_c_proj_bias_shape(768);
    const std::string &attn_c_proj_bias_path = "./input_res/p_attn_bias.npy";
    BITensor attn_c_proj_bias = utils::create_type_tensor(attn_c_proj_bias_path, attn_c_proj_bias_shape,
                                                          BIDataType::F16);

    PermutationVector q_perm{0, 2, 1, 3};
    PermutationVector k_perm{2, 0, 1, 3};
    PermutationVector qkv_o_perm{0, 2, 1, 3};

    // MLP 层权重
    const std::string &gpt_ln_2_path = "./intent_res/ln_2_weight_1.npy";
    BITensor gpt_ln_2_weight = OperatorTest::create_norm_input(std::vector<int>{768}, gpt_ln_2_path);
    const std::string &ln_2_bias_path = "./intent_res/ln_2_bias_1.npy";
    BITensor gpt_ln_2_bias = utils::create_type_tensor(ln_2_bias_path,
                                                       gpt_ln_1_weight_shape,
                                                       BIDataType::F16);
    // 初始化fc_weights的权重
    const std::string &c_fc_weights_path =
            "./input_res/reordered_c_fc_weights.npy";
    BITensor c_fc_weights = utils::create_type_tensor(c_fc_weights_path, BITensorShape(3072, 768), BIDataType::F16);
    // 初始化fc_bias
    const std::string &c_fc_bias_path = "./input_res/c_fc_bias.npy";
    BITensor c_fc_bias = utils::create_type_tensor(c_fc_bias_path, BITensorShape(3072), BIDataType::F16);
    // 6. proj的权重
    const std::string &c_proj_path = "./input_res/c_proj_weights.npy";
    BITensor c_proj_weight = OperatorTest::create_norm_input(std::vector<int>{3072, 768}, c_proj_path);
    const std::string &c_proj_bias_path = "./input_res/c_proj_bias.npy";
    BITensor c_proj_bias = OperatorTest::create_norm_input(std::vector<int>{768}, c_proj_bias_path);
    const BIActivationLayerInfo act_info(BIActivationFunction::GELU);

    // 合并Config
    BIGPTGlobalConfig gpt_block_config;
    gpt_block_config.q_perm = q_perm;
    gpt_block_config.k_perm = k_perm;
    gpt_block_config.qkv_perm = qkv_o_perm;
    gpt_block_config.hidden_size = hidden_size;
    gpt_block_config.max_seq_len = max_seq;
    gpt_block_config.max_batch_size = max_batch_size;

    std::vector<BIGPTLayerConfig> layer_config{};
    layer_config.reserve(block_layer_num);
    BIGPTLayerConfig gpt_layer_config_1, gpt_layer_config_2;
    gpt_layer_config_1.ln_1_weight = &gpt_ln_1_weights;
    gpt_layer_config_1.ln_1_bias = &gpt_ln_1_bias;
    gpt_layer_config_1.c_attn_weights = &gpt_c_attn_weights;
    gpt_layer_config_1.c_attn_bias = &gpt_c_attn_bias;
    gpt_layer_config_1.o_attn_weights = &attn_c_proj_weights;
    gpt_layer_config_1.o_attn_bias = &attn_c_proj_bias;
    gpt_layer_config_1.fc_weights = &c_fc_weights;
    gpt_layer_config_1.fc_bias = &c_fc_bias;
    gpt_layer_config_1.ln_2_weight = &gpt_ln_2_weight;
    gpt_layer_config_1.ln_2_bias = &gpt_ln_2_bias;
    gpt_layer_config_1.proj_weights = &c_proj_weight;
    gpt_layer_config_1.proj_bias = &c_proj_bias;
    gpt_layer_config_1.act_info = act_info;
    gpt_layer_config_1.layer_idx = 0;
    layer_config.push_back(gpt_layer_config_1);
    //
    // gpt_layer_config_2.ln_1_weight = &gpt_ln_1_weights;
    // gpt_layer_config_2.c_attn_weights = &gpt_c_attn_weights;
    // gpt_layer_config_2.c_attn_bias = &gpt_c_attn_bias;
    // gpt_layer_config_2.o_attn_weights = &attn_c_proj_weights;
    // gpt_layer_config_2.o_attn_bias = &attn_c_proj_bias;
    // gpt_layer_config_2.fc_weights = &c_fc_weights;
    // gpt_layer_config_2.fc_bias = &c_fc_bias;
    // gpt_layer_config_2.ln_2_weight = &gpt_ln_2_weight;
    // gpt_layer_config_2.proj_weights = &c_proj_weight;
    // gpt_layer_config_2.proj_bias = &c_proj_bias;
    // gpt_layer_config_2.act_info = act_info;
    // gpt_layer_config_2.layer_idx = 1;
    // layer_config.push_back(gpt_layer_config_2);
    //
    BITensor multi_gpt_o_t, sub_multi_gpt_o_t;
    multi_gpt_o_t.allocator()->init(BITensorInfo(gpt_o_tensor_shape, 1, BIDataType::F16));
    multi_gpt_o_t.allocator()->allocate();
    sub_multi_gpt_o_t.allocator()->init(*multi_gpt_o_t.allocator(), sub_gpt_o_t_info);


    BINEIntentMGPTBlock gpt_multi_block_layer;
    gpt_multi_block_layer.configure(&sub_wte_o_tensor,
                                    layer_config,
                                    gpt_block_config,
                                    batch_size,
                                    seq_len,
                                    &sub_multi_gpt_o_t);
    gpt_multi_block_layer.run();
    //
    // // OperatorTest::print_tensor(sub_multi_gpt_o_t, "output");
    //
    // // 6. 进行后续的GPT结果解码(先进行RMSNorm操作)
    // BITensor ln_f_o_tensor, sub_ln_f_o_tensor;
    // BITensor ln_f_weights = OperatorTest::create_norm_input(std::vector{768},
    //                                                            "./input_res/mlp_after_rms_gamma.npy");
    // ln_f_o_tensor.allocator()->init(BITensorInfo(gpt_o_tensor_shape, 1, BIDataType::F16));
    // ln_f_o_tensor.allocator()->allocate();
    // sub_ln_f_o_tensor.allocator()->init(*ln_f_o_tensor.allocator(), sub_gpt_o_t_info);
    // BINERMSNormLayer ln_f_layer;
    // ln_f_layer.configure(&sub_multi_gpt_o_t, &ln_f_weights, &sub_ln_f_o_tensor);
    // ln_f_layer.run();
    //
    // // 7. 进行LMHead操作
    // BITensor lm_head_weights = OperatorTest::create_norm_input(std::vector{768, 6003},
    //                                                               "./input_res/lm_head_weights.npy");
    // BITensor lm_head_output;
    // lm_head_output.allocator()->init(BITensorInfo(BITensorShape(6003, 1, max_batch_size), 1, BIDataType::F16));
    // lm_head_output.allocator()->allocate();
    // BITensor sub_lm_head_output;
    // BITensorInfo sub_lm_head_output_info = BITensorInfo(BITensorShape(6003, 1, batch_size), 1, BIDataType::F16);
    // sub_lm_head_output_info.set_format(Format::F16);
    // sub_lm_head_output.allocator()->init(*lm_head_output.allocator(), sub_lm_head_output_info);
    // GEMMInfo gemm_info = GEMMInfo(false,
    //                               false,
    //                               true,
    //                               false,
    //                               false,
    //                               false,
    //                               BIGEMMLowpOutputStageInfo(),
    //                               false, true, false,
    //                               BIActivationLayerInfo(), false, BIWeightFormat::UNSPECIFIED, false);
    // BINEGEMM lm_head_layer;
    // lm_head_layer.configure(&sub_ln_f_o_tensor, &lm_head_weights, nullptr, &sub_lm_head_output, 1.0f, 1.0f, gemm_info);
    // lm_head_layer.run();
    //
    // // 8. 最后生成结果
    // BITensor ids;
    // ids.allocator()->init(BITensorInfo(BITensorShape(1, max_batch_size), 1, BIDataType::S32));
    // ids.allocator()->allocate();
    // BITensor sub_ids;
    // BITensorInfo sub_ids_info = BITensorInfo(BITensorShape(1, batch_size), 1, BIDataType::S32);
    // sub_ids_info.set_format(Format::S32);
    // sub_ids.allocator()->init(*ids.allocator(), sub_ids_info);
    //
    // BINEArgMinMaxLayer arg_minmax_layer;
    // arg_minmax_layer.configure(&sub_lm_head_output, 0, &sub_ids, BIReductionOperation::ARG_IDX_MAX);
    // arg_minmax_layer.run();
    // std::vector<int> infos{};
    // std::vector<float> score{};
    // OperatorTest::concat_tensor(sub_ids, output_ids);
    // OperatorTest::get_s32_val(sub_ids, infos);
    // OperatorTest::get_index_val(sub_lm_head_output, infos, score);
    // scores.push_back(score[0]);
    //
    // OperatorTest::print_output_info(output_ids);
    // OperatorTest::print_output_info(scores);
    //
    // // ============================== 进行动态更新  ========================================
    // // 1. 先更新batch size和KV Cache
    // batch_size = 20;
    // seq_len = 2;
    // input_info.set_tensor_shape(BITensorShape(seq_len, batch_size));
    // sub_input_tensor.allocator()->init(*input_tensor.allocator(), input_info);
    // indices_data = {0, 3};
    // OperatorTest::fill_tensor_with_repeat_arr(input_ids, batch_size, indices_data);
    // OperatorTest::fill_tensor_val_with_arr(input_tensor, input_ids);
    // sub_gather_info.set_tensor_shape(BITensorShape(hidden_size, seq_len, batch_size));
    // sub_gather_o_tensor.allocator()->init(*gather_o_tensor.allocator(), sub_gather_info);
    // sub_wte_o_tensor.allocator()->init(*wte_o_tensor.allocator(), sub_gather_info);
    // sub_add_weight_info.set_tensor_shape(BITensorShape(hidden_size, seq_len));
    // sub_add_weight.allocator()->init(*add_wte_weight.allocator(), sub_add_weight_info);
    // sub_gpt_o_t_info.set_tensor_shape(BITensorShape(hidden_size, 1, batch_size));
    // sub_multi_gpt_o_t.allocator()->init(*multi_gpt_o_t.allocator(), sub_gpt_o_t_info);
    // sub_ln_f_o_tensor.allocator()->init(*ln_f_o_tensor.allocator(), sub_gpt_o_t_info);
    // sub_lm_head_output_info.set_tensor_shape(BITensorShape(6003, 1, batch_size));
    // sub_lm_head_output.allocator()->init(*lm_head_output.allocator(), sub_lm_head_output_info);
    // sub_ids_info.set_tensor_shape(BITensorShape(1, batch_size));
    // sub_ids.allocator()->init(*ids.allocator(), sub_ids_info);
    // gather_layer.dynamic_configure(&sub_input_tensor, &sub_gather_o_tensor);
    // wte_layer.dynamic_configure(&sub_gather_o_tensor, &sub_add_weight, true);
    // gpt_multi_block_layer.dynamic_configure(&sub_wte_o_tensor, seq_len, batch_size);
    // ln_f_layer.dynamic_configure(&sub_multi_gpt_o_t);
    // lm_head_layer.dynamic_configure();
    // arg_minmax_layer.configure(&sub_lm_head_output, 0, &sub_ids, BIReductionOperation::ARG_IDX_MAX);
    // gather_layer.run();
    // wte_layer.run();
    // gpt_multi_block_layer.run();
    // ln_f_layer.run();
    // lm_head_layer.run();
    // arg_minmax_layer.run();
    // OperatorTest::concat_tensor(sub_ids, output_ids);
    // OperatorTest::get_s32_val(sub_ids, infos);
    // OperatorTest::get_index_val(sub_lm_head_output, infos, score);
    // scores.push_back(score[0]);
    //
    // OperatorTest::print_output_info(output_ids);
    // OperatorTest::print_output_info(scores);
    // OperatorTest::print_tensor(sub_ids, "sub_ids");
    // // OperatorTest::print_tensor(sub_lm_head_output, "sub_lm_head");
    //
    // seq_len = 3;
    // input_info.set_tensor_shape(BITensorShape(seq_len, batch_size));
    // sub_input_tensor.allocator()->init(*input_tensor.allocator(), input_info);
    // indices_data = {0, 3, 4};
    // OperatorTest::fill_tensor_with_repeat_arr(input_ids, batch_size, indices_data);
    // OperatorTest::fill_tensor_val_with_arr(input_tensor, input_ids);
    // sub_gather_info.set_tensor_shape(BITensorShape(hidden_size, seq_len, batch_size));
    // sub_gather_o_tensor.allocator()->init(*gather_o_tensor.allocator(), sub_gather_info);
    // sub_wte_o_tensor.allocator()->init(*wte_o_tensor.allocator(), sub_gather_info);
    // sub_add_weight_info.set_tensor_shape(BITensorShape(hidden_size, seq_len));
    // sub_add_weight.allocator()->init(*add_wte_weight.allocator(), sub_add_weight_info);
    // sub_gpt_o_t_info.set_tensor_shape(BITensorShape(hidden_size, 1, batch_size));
    // sub_multi_gpt_o_t.allocator()->init(*multi_gpt_o_t.allocator(), sub_gpt_o_t_info);
    // sub_ln_f_o_tensor.allocator()->init(*ln_f_o_tensor.allocator(), sub_gpt_o_t_info);
    // sub_lm_head_output_info.set_tensor_shape(BITensorShape(6003, 1, batch_size));
    // sub_lm_head_output.allocator()->init(*lm_head_output.allocator(), sub_lm_head_output_info);
    // sub_ids_info.set_tensor_shape(BITensorShape(1, batch_size));
    // sub_ids.allocator()->init(*ids.allocator(), sub_ids_info);
    // gather_layer.dynamic_configure(&sub_input_tensor, &sub_gather_o_tensor);
    // wte_layer.dynamic_configure(&sub_gather_o_tensor, &sub_add_weight, true);
    // gpt_multi_block_layer.dynamic_configure(&sub_wte_o_tensor, seq_len, batch_size);
    // ln_f_layer.dynamic_configure(&sub_multi_gpt_o_t);
    // lm_head_layer.dynamic_configure();
    // arg_minmax_layer.configure(&sub_lm_head_output, 0, &sub_ids, BIReductionOperation::ARG_IDX_MAX);
    // gather_layer.run();
    // wte_layer.run();
    // gpt_multi_block_layer.run();
    // ln_f_layer.run();
    // lm_head_layer.run();
    // arg_minmax_layer.run();
    // OperatorTest::concat_tensor(sub_ids, output_ids);
    // OperatorTest::get_s32_val(sub_ids, infos);
    // OperatorTest::get_index_val(sub_lm_head_output, infos, score);
    // scores.push_back(score[0]);
    //
    // OperatorTest::print_output_info(output_ids);
    // OperatorTest::print_output_info(scores);
}
