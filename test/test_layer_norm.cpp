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
