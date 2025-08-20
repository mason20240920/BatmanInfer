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
#include <kv_cache_manager/bi_kv_cache_manager.hpp>

#include "cpu/kernels/assembly/arm_gemm_local.hpp"
#include "runtime/neon/bi_ne_scheduler.hpp"
#include "runtime/omp/bi_imp_scheduler.hpp"

namespace BugsTest {
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


TEST(BugTest, TestBugTestHold) {
    std::vector<size_t> avail_lens{1};
    using namespace BatmanInfer;
    // 输出的ids
    std::vector<int> output_ids{};
    std::vector<float> scores{};
    // 需要复制传入数组的值
    std::vector<uint32_t> input_ids{};
    // 1. 先初始化KV Cache的数据
    constexpr int num_head = 12;
    constexpr int head_dim = 64;
    constexpr int max_seq = 16;
    constexpr int hidden_size = 768;
    constexpr int max_batch_size = 20;
    constexpr int kv_manager_num = 512;
    constexpr int kv_mem_size = num_head * head_dim * sizeof(int8_t) + num_head * head_dim * sizeof(float16_t);
    // 每次存取的时候会存储K和V
    std::vector<unsigned int> kv_block_ids;
    // ======================== 0. 拷贝初始化占位符 ========================
    KVCacheManager::initialize(kv_manager_num, kv_mem_size, max_seq, 6);
    BITensorShape eos_tensor_shape(head_dim, num_head, max_seq);
    const std::string &eos_k_tensor_path = "./quant_res/eos_k_smooth_o.npy";
    BITensor eos_k_weights = utils::create_type_tensor(eos_k_tensor_path, eos_tensor_shape, BIDataType::F16);
    const std::string &eos_v_tensor_path = "./quant_res/eos_v_smooth_o.npy";
    BITensor eos_v_weights = utils::create_type_tensor(eos_v_tensor_path, eos_tensor_shape, BIDataType::QASYMM8_SIGNED);
    KVCacheManager::getInstance().memcpy_init_eos_buffer(eos_k_weights.buffer(), max_seq - 1, max_seq, true, true);
    KVCacheManager::getInstance().memcpy_init_eos_buffer(eos_v_weights.buffer(), max_seq - 1, max_seq, false, true);

    const auto root_id = KVCacheManager::getInstance().root_id();
    BIScheduler::get().set_num_threads(std::thread::hardware_concurrency());

    int batch_size = 1;
    int seq_len = 1;
    // ======================== 1. 初始化一个最大input算子 ========================
    BITensor input_tensor;
    BITensorShape input_tensor_shape(max_seq, max_batch_size);
    input_tensor.allocator()->init(BITensorInfo(input_tensor_shape, 1, BIDataType::U32));
    input_tensor.allocator()->allocate();
    BITensor sub_i_t;
    BITensorShape sub_i_shape(seq_len, batch_size);
    BITensorInfo sub_input_info(sub_i_shape, 1, BIDataType::U32);
    sub_input_info.set_format(Format::U32);
    sub_i_t.allocator()->init(*input_tensor.allocator(), sub_input_info);
    std::vector<uint32_t> indices_data{0};
    BugsTest::fill_tensor_val_with_arr(sub_i_t, indices_data);


    // ======================== 2. WTE位置编码的权重 ========================
    BITensorShape wte_weight_shape(hidden_size, 6003);
    const std::string &wte_weight_path = "./quant_res/transformer_wte_weight.npy";
    BITensor wte_weight = utils::create_type_tensor(wte_weight_path, wte_weight_shape, BIDataType::F16);

    BITensor wte_o_tensor;
    BITensorShape wte_o_tensor_shape(hidden_size, max_seq, max_batch_size);
    wte_o_tensor.allocator()->init(BITensorInfo(wte_o_tensor_shape, 1, BIDataType::F16));
    wte_o_tensor.allocator()->allocate();
    BITensor sub_wte_o_tensor;
    BITensorShape sub_wte_o_tensor_shape(hidden_size, seq_len, batch_size);
    BITensorInfo sub_wte_o_tensor_info(sub_wte_o_tensor_shape, 1, BIDataType::F16);
    sub_wte_o_tensor_info.set_format(Format::F16);
    sub_wte_o_tensor.allocator()->init(*wte_o_tensor.allocator(), sub_wte_o_tensor_info);

    BINEGather wte_layer;
    wte_layer.configure(&wte_weight, &sub_i_t, &sub_wte_o_tensor, 1);
    wte_layer.run();

    // ======================== 3. WPE编码相加 =====================
    BITensorShape add_wpe_weight_shape(hidden_size, max_seq);
    const std::string &add_wpe_weight_path = "./quant_res/add_wte_weights.npy";
    BITensor add_wte_weight = utils::create_type_tensor(add_wpe_weight_path,
                                                        add_wpe_weight_shape,
                                                        BIDataType::F16);
    BITensor sub_wpe_weight;
    BITensorShape sub_wpe_weight_shape(hidden_size, seq_len);
    BITensorInfo sub_wpe_weight_info(sub_wpe_weight_shape, 1, BIDataType::F16);
    sub_wpe_weight_info.set_format(Format::F16);
    sub_wpe_weight.allocator()->init(*add_wte_weight.allocator(), sub_wpe_weight_info);

    BITensor wpe_o_tensor;
    wpe_o_tensor.allocator()->init(BITensorInfo(wte_o_tensor_shape, 1, BIDataType::F16));
    wpe_o_tensor.allocator()->allocate();
    BITensor sub_wpe_o_tensor;
    sub_wpe_o_tensor.allocator()->init(*wpe_o_tensor.allocator(), sub_wte_o_tensor_info);

    BINEArithmeticAddition wpe_layer;
    wpe_layer.configure(&sub_wte_o_tensor, &sub_wpe_weight, &sub_wpe_o_tensor, BIConvertPolicy::SATURATE);
    wpe_layer.run();

    // ======================== 4. 进行Split切割 =====================
    BITensorShape kv_attn_o_shape = BITensorShape(hidden_size, 1, max_batch_size);
    BITensorShape sub_kv_attn_o_shape = BITensorShape(hidden_size, 1, batch_size);
    BITensorInfo sub_kv_attn_o_info(sub_kv_attn_o_shape, 1, BIDataType::F16);
    sub_kv_attn_o_info.set_format(Format::F16);
    BITensor kv_split_o_tensor, sub_kv_split_o_tensor;
    kv_split_o_tensor.allocator()->init(BITensorInfo(kv_attn_o_shape, 1, BIDataType::F16));
    kv_split_o_tensor.allocator()->allocate();
    sub_kv_split_o_tensor.allocator()->init(*kv_split_o_tensor.allocator(), sub_kv_attn_o_info);
    BIITensorPack pack;
    pack.add_tensor(ACL_SRC, &sub_wpe_o_tensor);
    pack.add_tensor(ACL_DST, &sub_kv_split_o_tensor);
    BINEScheduler::get().schedule_kv_split(pack, avail_lens);

    // ======================== 5. AttentionLowp算子计算 =====================
    BITensorShape attn_ln_1_shape(768);
    const std::string &attn_ln_1_path = "./quant_res/attn_gamma_weights.npy";
    BITensor attn_ln_1_weight = utils::create_type_tensor(attn_ln_1_path,
                                                          attn_ln_1_shape,
                                                          BIDataType::F16);
    const std::string &eos_weights_path = "./quant_res/eos_q_smooth_o.npy";
    constexpr float attn_i_scale = 0.007593675220713896;
    constexpr int attn_i_zp = 27;
    BITensorShape attn_qkv_weights_shape(hidden_size * 3, hidden_size);
    const std::string &attn_qkv_weights_path = "./quant_res/c_attn_weights.npy";
    BITensor attn_qkv_weights = utils::create_type_tensor(attn_qkv_weights_path,
                                                          attn_qkv_weights_shape,
                                                          BIDataType::QSYMM8_PER_CHANNEL);
    std::ifstream attn_qkv_weights_scale_file("./quant_res/c_attn_scales.txt");
    float value;
    std::vector<float> attn_qkv_weights_scales;
    while (attn_qkv_weights_scale_file >> value) {
        attn_qkv_weights_scales.push_back(value);
    }
    BIQuantizationInfo attn_qkv_weights_qinfo(attn_qkv_weights_scales);
    attn_qkv_weights.info()->set_quantization_info(attn_qkv_weights_qinfo);
    BITensorShape attn_qkv_bias_shape(2304);
    const std::string &attn_qkv_bias_path = "./quant_res/c_attn_bias.npy";
    BITensor attn_qkv_bias = utils::create_type_tensor(attn_qkv_bias_path, attn_qkv_bias_shape,
                                                       BIDataType::S32);
    BITensorShape attn_c_proj_weights_shape(768, 768);
    const std::string &attn_c_proj_weights_path = "./quant_res/p_attn_weights.npy";
    BITensor attn_c_proj_weights = utils::create_type_tensor(attn_c_proj_weights_path, attn_c_proj_weights_shape,
                                                             BIDataType::F16);
    BITensorShape attn_c_proj_bias_shape(768);
    const std::string &attn_c_proj_bias_path = "./quant_res/p_attn_bias.npy";
    BITensor attn_c_proj_bias = utils::create_type_tensor(attn_c_proj_bias_path, attn_c_proj_bias_shape,
                                                          BIDataType::F16);
    constexpr float attn_gemm_o_scale = 0.06920977199778837f;
    constexpr int attn_gemm_o_zero = 2;
    constexpr float query_scale = 0.03962391498042088f;
    constexpr int query_zp = 0;
    constexpr float value_scale = 0.06920977199778837f;
    constexpr int value_zp = 2;
    constexpr float key_scale = 0.03963086932313208f;
    constexpr int key_zp = 5;
    constexpr float softmax_q_scale = 0.00392156862745098f;
    constexpr int softmax_zp = -128;
    constexpr float proj_in_scale = 0.06920977199778837f;
    constexpr int proj_in_zp = 2;
    BINEAttentionLowpLayer attn_lowp_layer;

    PermutationVector q_perm{0, 2, 1, 3};
    PermutationVector k_perm{2, 0, 1, 3};
    PermutationVector qkv_o_perm{0, 2, 1, 3};
    BITensor attn_o_tensor, sub_attn_o_tensor;
    attn_o_tensor.allocator()->init(BITensorInfo(kv_attn_o_shape, 1, BIDataType::F16));
    attn_o_tensor.allocator()->allocate();
    sub_attn_o_tensor.allocator()->init(*attn_o_tensor.allocator(), sub_kv_attn_o_info);
    BITensor eos_weights = utils::create_type_tensor(eos_weights_path, BITensorShape(64, 12, 16), BIDataType::F16);
    // attn_lowp_layer.configure(&sub_kv_split_o_tensor,
    //                           &attn_ln_1_weight,
    //                           &attn_qkv_weights,
    //                           &attn_qkv_bias,
    //                           &attn_c_proj_weights,
    //                           &attn_c_proj_bias,
    //                           &eos_weights,
    //                           attn_i_scale,
    //                           attn_i_zp,
    //                           attn_gemm_o_scale,
    //                           attn_gemm_o_zero,
    //                           query_scale,
    //                           query_zp,
    //                           value_scale,
    //                           value_zp,
    //                           key_scale,
    //                           key_zp,
    //                           softmax_q_scale,
    //                           softmax_zp,
    //                           proj_in_scale,
    //                           proj_in_zp,
    //                           q_perm,
    //                           k_perm,
    //                           qkv_o_perm,
    //                           768,
    //                           max_seq,
    //                           20,
    //                           &sub_attn_o_tensor);
    attn_lowp_layer.set_avail_lens(&avail_lens);
    attn_lowp_layer.run();
    attn_lowp_layer.get_kv_block_ids(kv_block_ids);

    // ======================== 6. AttentionLowp输出和输入进行归一化 =====================
    BITensor con_attn_mlp_tensor, sub_con_attn_mlp_tensor;
    BITensorShape con_attn_mlp_shape(hidden_size, 1, max_batch_size);
    con_attn_mlp_tensor.allocator()->init(BITensorInfo(con_attn_mlp_shape, 1, BIDataType::F16));
    con_attn_mlp_tensor.allocator()->allocate();
    sub_con_attn_mlp_tensor.allocator()->init(*con_attn_mlp_tensor.allocator(), sub_kv_attn_o_info);
    BINEArithmeticAddition con_add_1_layer;
    con_add_1_layer.configure(&sub_kv_split_o_tensor,
                              &sub_attn_o_tensor,
                              &sub_con_attn_mlp_tensor,
                              BIConvertPolicy::SATURATE);
    con_add_1_layer.run();

    // ======================== 7. 进行MLP层的计算 =====================
    BINEMLPLayer _mlp_layer;
    const std::string &ln_2_path = "./quant_res/mlp_rms_gamma.npy";
    BITensor ln_2_weight = BugsTest::create_norm_input(std::vector<int>{hidden_size}, ln_2_path);
    const std::string &c_fc_weights_path = "./quant_res/reordered_c_fc_weights.npy";
    std::vector<float> c_fc_weights_scales;
    BIQuantizationInfo c_fc_weight_qinfo;
    std::ifstream c_fc_weights_scale_file("./quant_res/c_fc_scales.txt");
    float s_value;
    while (c_fc_weights_scale_file >> s_value) {
        c_fc_weights_scales.push_back(s_value);
    }
    BITensor c_fc_weights = BugsTest::create_per_channel(c_fc_weights_scales, std::vector{hidden_size, 3072},
                                                                c_fc_weight_qinfo, c_fc_weights_path);
    const std::string &c_fc_bias_path = "./quant_res/c_fc_bias.npy";
    BITensor c_fc_bias = BugsTest::create_norm_bias(3072, c_fc_bias_path);
    BITensor mlp_output, sub_mlp_output;
    mlp_output.allocator()->init(BITensorInfo(con_attn_mlp_shape, 1, BIDataType::F16));
    mlp_output.allocator()->allocate();
    sub_mlp_output.allocator()->init(*mlp_output.allocator(), sub_kv_attn_o_info);
    const std::string &c_proj_path = "./quant_res/c_proj_weights.npy";
    BITensor c_proj_weight = BugsTest::create_norm_input(std::vector<int>{3072, hidden_size}, c_proj_path);
    const std::string &c_proj_bias_path = "./quant_res/c_proj_bias.npy";
    BITensor c_proj_bias = BugsTest::create_norm_input(std::vector<int>{hidden_size}, c_proj_bias_path);
    float fc1_input_scale = 0.007606653839934106f;
    int fc1_input_zero_point = 14;
    float fc1_output_scale = 0.2595925574209176f;
    int fc1_output_zero_point = -10;
    float gelu_output_scale = 0.14033655150263918f;
    int gelu_output_zero_point = -127;
    _mlp_layer.configure(&sub_con_attn_mlp_tensor,
                         fc1_input_scale,
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
                         &ln_2_weight,
                         &sub_mlp_output,
                         max_batch_size,
                         1);
    _mlp_layer.run();

    // ======================== 8. MLP层输出和输入进行二次归一化 =====================
    BITensor con_mlp_in_tensor, sub_con_mlp_in_tensor;
    con_mlp_in_tensor.allocator()->init(BITensorInfo(con_attn_mlp_shape, 1, BIDataType::F16));
    con_mlp_in_tensor.allocator()->allocate();
    sub_con_mlp_in_tensor.allocator()->init(*con_mlp_in_tensor.allocator(), sub_kv_attn_o_info);
    BINEArithmeticAddition con_add_2_layer;
    con_add_2_layer.configure(&sub_con_attn_mlp_tensor,
                              &sub_mlp_output,
                              &sub_con_mlp_in_tensor,
                              BIConvertPolicy::SATURATE);
    con_add_2_layer.run();

    // ======================== 9. 进行输出后的RMSNorm =====================
    BITensor ln_f_weights = BugsTest::create_norm_input(std::vector{hidden_size},
                                                               "./quant_res/mlp_after_rms_gamma.npy");
    BITensor ln_f_output, sub_ln_f_output;
    ln_f_output.allocator()->init(BITensorInfo(con_attn_mlp_shape, 1, BIDataType::F16));
    ln_f_output.allocator()->allocate();
    sub_ln_f_output.allocator()->init(*ln_f_output.allocator(), sub_kv_attn_o_info);
    BINERMSNormLayer ln_f_layer;
    ln_f_layer.configure(&sub_con_mlp_in_tensor, &ln_f_weights, &sub_ln_f_output);
    ln_f_layer.run();

    // ======================== 10. 进行LM Head头操作 =====================
    BITensor lm_head_weights = BugsTest::create_norm_input(std::vector{hidden_size, 6003},
                                                                  "./quant_res/lm_head_weights.npy");
    BITensor lm_head_output, sub_lm_head_output;
    lm_head_output.allocator()->init(BITensorInfo(BITensorShape(6003, 1, max_batch_size), 1, BIDataType::F16));
    lm_head_output.allocator()->allocate();
    BITensorInfo sub_lm_head_output_info = BITensorInfo(BITensorShape(6003, 1, batch_size), 1, BIDataType::F16);
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
    lm_head_layer.configure(&sub_ln_f_output, &lm_head_weights, nullptr, &sub_lm_head_output, 1.0f, 1.0f, gemm_info);
    lm_head_layer.run();

    // ======================== 11. 最终的结果id过滤 =====================
    BITensor ids, sub_ids;
    ids.allocator()->init(BITensorInfo(BITensorShape(1, max_batch_size), 1, BIDataType::S32));
    ids.allocator()->allocate();
    BITensorInfo sub_ids_info = BITensorInfo(BITensorShape(1, batch_size), 1, BIDataType::S32);
    sub_ids_info.set_format(Format::S32);
    sub_ids.allocator()->init(*ids.allocator(), sub_ids_info);

    BINEArgMinMaxLayer arg_minmax_layer;
    arg_minmax_layer.configure(&sub_lm_head_output, 0, &sub_ids, BIReductionOperation::ARG_IDX_MAX);
    arg_minmax_layer.run();

    std::vector<int> infos{};
    std::vector<float> score{};
    BugsTest::concat_tensor(sub_ids, output_ids);
    BugsTest::get_s32_val(sub_ids, infos);
    BugsTest::get_index_val(sub_lm_head_output, infos, score);
    scores.push_back(score[0]);

    BugsTest::print_output_info(output_ids);
    BugsTest::print_output_info(scores);


    /*
     * ============================================================
     * ======================== 第二轮的计算 ========================
     * ============================================================
    */
    batch_size = 20;
    std::vector<std::vector<unsigned int> > inp_map{};
    for (int i = 0; i < batch_size; i++) {
        inp_map.push_back({kv_block_ids[0], 1});
    }
    avail_lens = {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
    seq_len = 2;
    sub_i_shape = BITensorShape(seq_len, batch_size);
    sub_input_info.set_tensor_shape(sub_i_shape);
    sub_i_t.allocator()->init(*input_tensor.allocator(), sub_input_info);
    indices_data = {
        0, 3, 0, 4, 0, 5,
        0, 3, 0, 4, 0, 5,
        0, 3, 0, 4, 0, 5,
        0, 3, 0, 4, 0, 5,
        0, 3, 0, 4, 0, 5,
        0, 3, 0, 4, 0, 5,
        0, 3, 0, 4
    };
    BugsTest::fill_tensor_val_with_arr(sub_i_t, indices_data);
    sub_wte_o_tensor_shape = BITensorShape(hidden_size, seq_len, batch_size);
    sub_kv_attn_o_shape = BITensorShape(hidden_size, 1, batch_size);
    sub_wte_o_tensor_info.set_tensor_shape(sub_wte_o_tensor_shape);
    sub_kv_attn_o_info.set_tensor_shape(sub_kv_attn_o_shape);
    sub_wte_o_tensor.allocator()->init(*wte_o_tensor.allocator(), sub_wte_o_tensor_info);
    sub_wpe_o_tensor.allocator()->init(*wpe_o_tensor.allocator(), sub_wte_o_tensor_info);
    sub_kv_split_o_tensor.allocator()->init(*kv_split_o_tensor.allocator(), sub_kv_attn_o_info);
    sub_wpe_weight_shape = BITensorShape(hidden_size, seq_len);
    sub_wpe_weight_info.set_tensor_shape(sub_wpe_weight_shape);
    sub_lm_head_output_info.set_tensor_shape(BITensorShape(6003, 1, batch_size));
    sub_wpe_weight.allocator()->init(*add_wte_weight.allocator(), sub_wpe_weight_info);
    sub_attn_o_tensor.allocator()->init(*attn_o_tensor.allocator(), sub_kv_attn_o_info);
    sub_con_attn_mlp_tensor.allocator()->init(*con_attn_mlp_tensor.allocator(), sub_kv_attn_o_info);
    sub_mlp_output.allocator()->init(*mlp_output.allocator(), sub_kv_attn_o_info);
    sub_con_mlp_in_tensor.allocator()->init(*con_mlp_in_tensor.allocator(), sub_kv_attn_o_info);
    sub_ln_f_output.allocator()->init(*ln_f_output.allocator(), sub_kv_attn_o_info);
    sub_lm_head_output.allocator()->init(*lm_head_output.allocator(), sub_lm_head_output_info);
    sub_ids_info.set_tensor_shape(BITensorShape(1, batch_size));
    sub_ids.allocator()->init(*ids.allocator(), sub_ids_info);
    wte_layer.dynamic_configure(&sub_i_t, &sub_wte_o_tensor);
    wpe_layer.dynamic_configure(&sub_wte_o_tensor, &sub_wpe_weight, true);
    attn_lowp_layer.dynamic_configure(&sub_kv_split_o_tensor, seq_len, batch_size, inp_map);
    con_add_1_layer.dynamic_configure(&sub_kv_split_o_tensor, &sub_attn_o_tensor, false);
    _mlp_layer.dynamic_configure(&sub_con_attn_mlp_tensor, batch_size);
    con_add_2_layer.dynamic_configure(&sub_mlp_output, &sub_con_mlp_in_tensor, false);
    ln_f_layer.dynamic_configure(&sub_con_mlp_in_tensor);
    lm_head_layer.dynamic_configure();
    arg_minmax_layer.configure(&sub_lm_head_output, 0, &sub_ids, BIReductionOperation::ARG_IDX_MAX);
    attn_lowp_layer.set_avail_lens(&avail_lens);
    wte_layer.run();
    wpe_layer.run();
    BINEScheduler::get().schedule_kv_split(pack, avail_lens);
    attn_lowp_layer.run();
    attn_lowp_layer.get_kv_block_ids(kv_block_ids);
    // BugsTest::print_tensor(sub_attn_o_tensor, "attn");
    con_add_1_layer.run();
    _mlp_layer.run();
    // BugsTest::print_tensor(sub_con_attn_mlp_tensor, "sub_con_attn_mlp_tensor");
    inp_map.clear();
    for (unsigned int &kv_block_id: kv_block_ids)
        inp_map.push_back({kv_block_id, 1});
    con_add_2_layer.run();
    ln_f_layer.run();
    lm_head_layer.run();
    arg_minmax_layer.run();

    BugsTest::concat_tensor(sub_ids, output_ids);
    BugsTest::get_s32_val(sub_ids, infos);
    BugsTest::get_index_val(sub_lm_head_output, infos, score);
    scores.push_back(score[0]);

    BugsTest::print_output_info(output_ids);
    BugsTest::print_output_info(scores);
    BugsTest::print_tensor(sub_ids, "sub_ids");
    // BugsTest::print_tensor(sub_lm_head_output, "scores");

    avail_lens = {3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3};
    seq_len = 3;
    sub_i_shape = BITensorShape(seq_len, batch_size);
    sub_input_info.set_tensor_shape(sub_i_shape);
    sub_i_t.allocator()->init(*input_tensor.allocator(), sub_input_info);
    indices_data = {
        0, 3, 4, 0, 4, 5, 0, 5, 6,
        0, 3, 4, 0, 4, 5, 0, 5, 6,
        0, 3, 4, 0, 4, 5, 0, 5, 6,
        0, 3, 4, 0, 4, 5, 0, 5, 6,
        0, 3, 4, 0, 4, 5, 0, 5, 6,
        0, 3, 4, 0, 4, 5, 0, 5, 6,
        0, 3, 4, 0, 4, 5
    };
    BugsTest::fill_tensor_val_with_arr(sub_i_t, indices_data);
    sub_wte_o_tensor_shape = BITensorShape(hidden_size, seq_len, batch_size);
    sub_kv_attn_o_shape = BITensorShape(hidden_size, 1, batch_size);
    sub_wte_o_tensor_info.set_tensor_shape(sub_wte_o_tensor_shape);
    sub_kv_attn_o_info.set_tensor_shape(sub_kv_attn_o_shape);
    sub_wte_o_tensor.allocator()->init(*wte_o_tensor.allocator(), sub_wte_o_tensor_info);
    sub_wpe_o_tensor.allocator()->init(*wpe_o_tensor.allocator(), sub_wte_o_tensor_info);
    sub_kv_split_o_tensor.allocator()->init(*kv_split_o_tensor.allocator(), sub_kv_attn_o_info);
    sub_wpe_weight_shape = BITensorShape(hidden_size, seq_len);
    sub_wpe_weight_info.set_tensor_shape(sub_wpe_weight_shape);
    sub_lm_head_output_info.set_tensor_shape(BITensorShape(6003, 1, batch_size));
    sub_wpe_weight.allocator()->init(*add_wte_weight.allocator(), sub_wpe_weight_info);
    sub_attn_o_tensor.allocator()->init(*attn_o_tensor.allocator(), sub_kv_attn_o_info);
    sub_con_attn_mlp_tensor.allocator()->init(*con_attn_mlp_tensor.allocator(), sub_kv_attn_o_info);
    sub_mlp_output.allocator()->init(*mlp_output.allocator(), sub_kv_attn_o_info);
    sub_con_mlp_in_tensor.allocator()->init(*con_mlp_in_tensor.allocator(), sub_kv_attn_o_info);
    sub_ln_f_output.allocator()->init(*ln_f_output.allocator(), sub_kv_attn_o_info);
    sub_lm_head_output.allocator()->init(*lm_head_output.allocator(), sub_lm_head_output_info);
    sub_ids_info.set_tensor_shape(BITensorShape(1, batch_size));
    sub_ids.allocator()->init(*ids.allocator(), sub_ids_info);
    wte_layer.dynamic_configure(&sub_i_t, &sub_wte_o_tensor);
    wpe_layer.dynamic_configure(&sub_wte_o_tensor, &sub_wpe_weight, true);
    attn_lowp_layer.dynamic_configure(&sub_kv_split_o_tensor, seq_len, batch_size, inp_map);
    con_add_1_layer.dynamic_configure(&sub_kv_split_o_tensor, &sub_attn_o_tensor, false);
    _mlp_layer.dynamic_configure(&sub_con_attn_mlp_tensor, batch_size);
    con_add_2_layer.dynamic_configure(&sub_mlp_output, &sub_con_mlp_in_tensor, false);
    ln_f_layer.dynamic_configure(&sub_con_mlp_in_tensor);
    lm_head_layer.dynamic_configure();
    arg_minmax_layer.configure(&sub_lm_head_output, 0, &sub_ids, BIReductionOperation::ARG_IDX_MAX);
    attn_lowp_layer.set_avail_lens(&avail_lens);
    wte_layer.run();
    wpe_layer.run();
    BINEScheduler::get().schedule_kv_split(pack, avail_lens);
    attn_lowp_layer.run();
    attn_lowp_layer.get_kv_block_ids(kv_block_ids);
    // BugsTest::print_tensor(sub_attn_o_tensor, "attn");
    con_add_1_layer.run();
    _mlp_layer.run();
    // BugsTest::print_tensor(sub_con_attn_mlp_tensor, "sub_con_attn_mlp_tensor");
    inp_map.clear();
    for (unsigned int &kv_block_id: kv_block_ids)
        inp_map.push_back({kv_block_id, 1});
    con_add_2_layer.run();
    ln_f_layer.run();
    lm_head_layer.run();
    arg_minmax_layer.run();

    BugsTest::concat_tensor(sub_ids, output_ids);
    BugsTest::get_s32_val(sub_ids, infos);
    BugsTest::get_index_val(sub_lm_head_output, infos, score);
    scores.push_back(score[0]);

    BugsTest::print_output_info(output_ids);
    BugsTest::print_output_info(scores);
    BugsTest::print_tensor(sub_ids, "sub_ids");
    // BugsTest::print_tensor(sub_lm_head_output, "scores");

    inp_map[0] = {527, 1};
    inp_map[3] = {527, 1};
    inp_map[6] = {527, 1};
    inp_map[9] = {527, 1};
    inp_map[12] = {527, 1};
    inp_map[15] = {527, 1};
    inp_map[18] = {527, 1};
    avail_lens = {
        2, 4, 4,
        2, 4, 4,
        2, 4, 4,
        2, 4, 4,
        2, 4, 4,
        2, 4, 4,
        2, 4
    };
    seq_len = 4;
    sub_i_shape = BITensorShape(seq_len, batch_size);
    sub_input_info.set_tensor_shape(sub_i_shape);
    sub_i_t.allocator()->init(*input_tensor.allocator(), sub_input_info);
    indices_data = {
        0, 5, 2, 2, 0, 4, 5, 6, 0, 5, 6, 7,
        0, 5, 2, 2, 0, 4, 5, 6, 0, 5, 6, 7,
        0, 5, 2, 2, 0, 4, 5, 6, 0, 5, 6, 7,
        0, 5, 2, 2, 0, 4, 5, 6, 0, 5, 6, 7,
        0, 5, 2, 2, 0, 4, 5, 6, 0, 5, 6, 7,
        0, 5, 2, 2, 0, 4, 5, 6, 0, 5, 6, 7,
        0, 5, 2, 2, 0, 4, 5, 6
    };
    BugsTest::fill_tensor_val_with_arr(sub_i_t, indices_data);
    sub_wte_o_tensor_shape = BITensorShape(hidden_size, seq_len, batch_size);
    sub_kv_attn_o_shape = BITensorShape(hidden_size, 1, batch_size);
    sub_wte_o_tensor_info.set_tensor_shape(sub_wte_o_tensor_shape);
    sub_kv_attn_o_info.set_tensor_shape(sub_kv_attn_o_shape);
    sub_wte_o_tensor.allocator()->init(*wte_o_tensor.allocator(), sub_wte_o_tensor_info);
    sub_wpe_o_tensor.allocator()->init(*wpe_o_tensor.allocator(), sub_wte_o_tensor_info);
    sub_kv_split_o_tensor.allocator()->init(*kv_split_o_tensor.allocator(), sub_kv_attn_o_info);
    sub_wpe_weight_shape = BITensorShape(hidden_size, seq_len);
    sub_wpe_weight_info.set_tensor_shape(sub_wpe_weight_shape);
    sub_lm_head_output_info.set_tensor_shape(BITensorShape(6003, 1, batch_size));
    sub_wpe_weight.allocator()->init(*add_wte_weight.allocator(), sub_wpe_weight_info);
    sub_attn_o_tensor.allocator()->init(*attn_o_tensor.allocator(), sub_kv_attn_o_info);
    sub_con_attn_mlp_tensor.allocator()->init(*con_attn_mlp_tensor.allocator(), sub_kv_attn_o_info);
    sub_mlp_output.allocator()->init(*mlp_output.allocator(), sub_kv_attn_o_info);
    sub_con_mlp_in_tensor.allocator()->init(*con_mlp_in_tensor.allocator(), sub_kv_attn_o_info);
    sub_ln_f_output.allocator()->init(*ln_f_output.allocator(), sub_kv_attn_o_info);
    sub_lm_head_output.allocator()->init(*lm_head_output.allocator(), sub_lm_head_output_info);
    sub_ids_info.set_tensor_shape(BITensorShape(1, batch_size));
    sub_ids.allocator()->init(*ids.allocator(), sub_ids_info);
    wte_layer.dynamic_configure(&sub_i_t, &sub_wte_o_tensor);
    wpe_layer.dynamic_configure(&sub_wte_o_tensor, &sub_wpe_weight, true);
    attn_lowp_layer.dynamic_configure(&sub_kv_split_o_tensor, seq_len, batch_size, inp_map);
    con_add_1_layer.dynamic_configure(&sub_kv_split_o_tensor, &sub_attn_o_tensor, false);
    _mlp_layer.dynamic_configure(&sub_con_attn_mlp_tensor, batch_size);
    con_add_2_layer.dynamic_configure(&sub_mlp_output, &sub_con_mlp_in_tensor, false);
    ln_f_layer.dynamic_configure(&sub_con_mlp_in_tensor);
    lm_head_layer.dynamic_configure();
    arg_minmax_layer.configure(&sub_lm_head_output, 0, &sub_ids, BIReductionOperation::ARG_IDX_MAX);
    attn_lowp_layer.set_avail_lens(&avail_lens);
    wte_layer.run();
    wpe_layer.run();
    BINEScheduler::get().schedule_kv_split(pack, avail_lens);
    attn_lowp_layer.run();
    attn_lowp_layer.get_kv_block_ids(kv_block_ids);
    // BugsTest::print_tensor(sub_attn_o_tensor, "attn");
    con_add_1_layer.run();
    _mlp_layer.run();
    // BugsTest::print_tensor(sub_con_attn_mlp_tensor, "sub_con_attn_mlp_tensor");
    inp_map.clear();
    for (unsigned int &kv_block_id: kv_block_ids)
        inp_map.push_back({kv_block_id, 1});
    con_add_2_layer.run();
    ln_f_layer.run();
    lm_head_layer.run();
    arg_minmax_layer.run();

    BugsTest::concat_tensor(sub_ids, output_ids);
    BugsTest::get_s32_val(sub_ids, infos);
    BugsTest::get_index_val(sub_lm_head_output, infos, score);
    scores.push_back(score[0]);

    BugsTest::print_output_info(output_ids);
    BugsTest::print_output_info(scores);
}