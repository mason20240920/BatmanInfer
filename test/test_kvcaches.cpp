//
// Created by Mason on 2025/2/21.
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

#include "runtime/neon/bi_ne_scheduler.hpp"
#include "runtime/omp/bi_imp_scheduler.hpp"

namespace KVCacheTestName {
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
}

using namespace BatmanInfer;

template<typename T>
void match_info(BITensor &tensor, const std::vector<T> &vec) {
    auto shape = tensor.info()->tensor_shape().total_size();
    auto tensor_data = reinterpret_cast<T *>(tensor.buffer());
    for (int i = 0; i < shape; i++) {
        tensor_data[i] = vec[i];
    }
}

TEST(KVCaches, NEGEMMINT8) {
    BIScheduler::get().set_num_threads(std::thread::hardware_concurrency());

    // 进行矩阵计算的KVCaches
    BIIOFormatInfo format;
    format.element_delim = ", "; // 元素之间用逗号分隔
    format.row_delim = "\n"; // 每行换行
    format.align_columns = true; // 对齐列

    int batch_size = 1;
    int sequence_len = 1;
    // 测试动态输入NEGEMM的过程
    BITensorShape tensor_a_shape(2, sequence_len, batch_size);
    BITensorShape tensor_b_shape(4, 2);
    BITensorShape tensor_bias_shape(4);
    BITensorShape tensor_d_shape(4, sequence_len, batch_size);

    BITensorInfo tensor_a_info(tensor_a_shape, 1, BIDataType::QASYMM8_SIGNED);
    BITensorInfo tensor_b_info(tensor_b_shape, 1, BIDataType::QASYMM8_SIGNED);
    BITensorInfo tensor_bias_info(tensor_bias_shape, 1, BIDataType::S32);
    BITensorInfo tensor_d_info(tensor_d_shape, 1, BIDataType::S32);
    BITensorInfo output_info(tensor_d_shape, 1, BIDataType::S32);

    BITensor tensor_a, tensor_b, bias, tensor_d, tensor_output;

    // 初始化
    tensor_a.allocator()->init(tensor_a_info);
    tensor_b.allocator()->init(tensor_b_info);
    bias.allocator()->init(tensor_bias_info);
    tensor_d.allocator()->init(tensor_d_info);
    tensor_output.allocator()->init(output_info);

    tensor_a.allocator()->allocate();
    tensor_b.allocator()->allocate();
    bias.allocator()->allocate();
    tensor_d.allocator()->allocate();
    tensor_output.allocator()->allocate();

    // 进行赋值
    std::vector<int8_t> data_a{1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<int8_t> data_b{1, 1, 1, 1, 1, 1, 1, 1};
    std::vector<int32_t> data_bias{3, 3, 3, 3};
    match_info(tensor_a, data_a);
    match_info(tensor_b, data_b);
    match_info(bias, data_bias);


    // 运行推理
    BINEGEMMLowpMatrixMultipleCore gemm;
    GEMMInfo gemm_info;
    gemm_info.set_fast_math(true);

    gemm.configure(&tensor_a, &tensor_b, nullptr, &tensor_d, gemm_info);

    BINEArithmeticAddition add;
    add.configure(&tensor_d, &bias, &tensor_output, BIConvertPolicy::WRAP);

    gemm.run();
    add.run();

    tensor_output.print(std::cout, format);
}

TEST(KVCaches, NEGEMMCaches) {
    BIScheduler::get().set_num_threads(std::thread::hardware_concurrency());
    // 进行矩阵计算的KVCaches
    BIIOFormatInfo format;
    format.element_delim = ", "; // 元素之间用逗号分隔
    format.row_delim = "\n"; // 每行换行
    format.align_columns = true; // 对齐列

    int batch_size = 1;
    int sequence_len = 16;
    int kv_one_len = 1;
    int head_num = 1;
    int head_dim = 2;
    // 定义输入和输出张量的形状
    BITensorShape shape_a(head_dim, kv_one_len, head_num, batch_size); // 左矩阵 (3x2)
    BITensorShape shape_b(sequence_len, head_dim, head_num, batch_size); // 右矩阵 (4x2)，需要转置为 (2x4)
    BITensorShape shape_c(sequence_len, kv_one_len, head_num, batch_size); // 输出矩阵 (4x3)

    // 创建输入和输出张量
    BITensor tensor_a, tensor_b, tensor_c;

    // 配置张量
    tensor_a.allocator()->init(BITensorInfo(shape_a, 1, BIDataType::F16));
    tensor_b.allocator()->init(BITensorInfo(shape_b, 1, BIDataType::F16));
    tensor_c.allocator()->init(BITensorInfo(shape_c, 1, BIDataType::F16));

    tensor_a.info()->set_are_values_constant(false);
    tensor_b.info()->set_are_values_constant(false);
    // 定义 MatMul 配置信息
    BIMatMulInfo matmul_info; // 不转置左矩阵，转置右矩阵
    matmul_info.adj_lhs(false).adj_rhs(false);
    BICpuMatMulSettings settings;
    settings.fast_math(true); // 启用快速数学模式
    //    settings.fixed_format(true);

    // 定义激活函数信息（可选）
    //    BIActivationLayerInfo act_info(BIActivationLayerInfo::ActivationFunction::RELU);

    // 创建 MatMul 操作对象
    BINEMatMul matmul;

    // 配置 MatMul 操作
    matmul.configure(&tensor_a, &tensor_b, &tensor_c, matmul_info, settings);

    // 分配内存
    tensor_a.allocator()->allocate();
    tensor_b.allocator()->allocate();
    tensor_c.allocator()->allocate();

    // 填充输入张量数据
    auto a_ptr = reinterpret_cast<float16_t *>(tensor_a.buffer());
    auto b_ptr = reinterpret_cast<float16_t *>(tensor_b.buffer());
    for (int i = 0; i < shape_a.total_size(); ++i) {
        a_ptr[i] = static_cast<float16_t>(i); // 示例数据
    }
    for (int i = 0; i < shape_b.total_size(); ++i) {
        b_ptr[i] = static_cast<float16_t>(1); // 示例数据
    }

    matmul.run();

    tensor_c.print(std::cout, format);

    std::cout << "=======" << std::endl;
    tensor_a.allocator()->free();
    tensor_b.allocator()->free();
    tensor_c.allocator()->free();
    batch_size = 2;
    shape_a = BITensorShape(head_dim, kv_one_len, head_num, batch_size); // 左矩阵 (3x2)
    shape_b = BITensorShape(sequence_len, head_dim, head_num, batch_size);
    shape_c = BITensorShape(sequence_len, kv_one_len, head_num, batch_size);
    tensor_a.allocator()->init(BITensorInfo(shape_a, 1, BIDataType::F16));
    tensor_b.allocator()->init(BITensorInfo(shape_b, 1, BIDataType::F16));
    tensor_c.allocator()->init(BITensorInfo(shape_c, 1, BIDataType::F16));
    tensor_b.allocator()->allocate();
    tensor_a.allocator()->allocate();
    tensor_c.allocator()->allocate();

    // 填充输入张量数据
    a_ptr = reinterpret_cast<float16_t *>(tensor_a.buffer());
    b_ptr = reinterpret_cast<float16_t *>(tensor_b.buffer());
    for (int i = 0; i < shape_a.total_size(); ++i) {
        a_ptr[i] = static_cast<float16_t>(i); // 示例数据
    }
    for (int i = 0; i < shape_b.total_size(); ++i) {
        b_ptr[i] = static_cast<float16_t>(1); // 示例数据
    }


    matmul.run();
    tensor_c.print(std::cout, format);
}

TEST(KVCaches, DynamicGemm) {
    BIScheduler::get().set_num_threads(std::thread::hardware_concurrency());
    // 进行矩阵计算的KVCaches
    BIIOFormatInfo format;
    format.element_delim = ", "; // 元素之间用逗号分隔
    format.row_delim = "\n"; // 每行换行
    format.align_columns = true; // 对齐列
    int batch_size = 1;
    int sequence_len = 2;
    // 测试动态输入NEGEMM的过程
    BITensorShape tensor_a_shape(2, sequence_len, batch_size);
    BITensorShape tensor_b_shape(4, 2);
    BITensorShape tensor_bias_shape(4);
    BITensorShape tensor_d_shape(4, sequence_len, batch_size);

    BITensorInfo tensor_a_info(tensor_a_shape, 1, BIDataType::F16);
    BITensorInfo tensor_b_info(tensor_b_shape, 1, BIDataType::F16);
    BITensorInfo tensor_bias_info(tensor_bias_shape, 1, BIDataType::F16);
    BITensorInfo tensor_d_info(tensor_d_shape, 1, BIDataType::F16);

    BITensor tensor_a, tensor_b, bias, tensor_d;

    // 初始化
    tensor_a.allocator()->init(tensor_a_info);
    tensor_b.allocator()->init(tensor_b_info);
    bias.allocator()->init(tensor_bias_info);
    tensor_d.allocator()->init(tensor_d_info);

    tensor_a.allocator()->allocate();
    tensor_b.allocator()->allocate();
    bias.allocator()->allocate();
    tensor_d.allocator()->allocate();

    // 进行赋值
    std::vector<float16_t> data_a{1, 2, 3, 4};
    std::vector<float16_t> data_b{1, 1, 1, 1, 1, 1, 1, 1};
    std::vector<float16_t> data_bias{3, 3, 3, 3};
    match_info(tensor_a, data_a);
    match_info(tensor_b, data_b);
    match_info(bias, data_bias);


    // 运行推理
    BINEGEMM gemm;
    GEMMInfo gemm_info;
    gemm_info.set_fast_math(true);

    gemm.configure(&tensor_a, &tensor_b, &bias, &tensor_d, 1.0f, 1.0f, gemm_info);

    gemm.run();

    tensor_d.print(std::cout, format);

    // 动态更新
    sequence_len = 3;
    batch_size = 2;
    tensor_a_shape = BITensorShape(2, sequence_len, batch_size);
    tensor_d_shape = BITensorShape(4, sequence_len, batch_size);

    tensor_a_info = BITensorInfo(tensor_a_shape, 1, BIDataType::F16);
    tensor_d_info = BITensorInfo(tensor_d_shape, 1, BIDataType::F16);

    // 初始化
    tensor_a.allocator()->init(tensor_a_info);
    tensor_d.allocator()->init(tensor_d_info);

    tensor_a.allocator()->allocate();
    tensor_d.allocator()->allocate();

    data_a = {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    match_info(tensor_a, data_a);
    gemm.run();

    tensor_d.print(std::cout, format);

    // 动态更新
    sequence_len = 4;
    batch_size = 3;
    tensor_a_shape = BITensorShape(2, sequence_len, batch_size);
    tensor_d_shape = BITensorShape(4, sequence_len, batch_size);

    tensor_a_info = BITensorInfo(tensor_a_shape, 1, BIDataType::F16);
    tensor_d_info = BITensorInfo(tensor_d_shape, 1, BIDataType::F16);

    // 初始化
    tensor_a.allocator()->init(tensor_a_info);
    tensor_d.allocator()->init(tensor_d_info);

    tensor_a.allocator()->allocate();
    tensor_d.allocator()->allocate();

    data_a = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    match_info(tensor_a, data_a);
    gemm.run();

    tensor_d.print(std::cout, format);
}

TEST(KVCacheGPT, GPT2KVCacheOrigin) {
    // 输出的ids
    std::vector<int> output_ids{};
    std::vector<float> scores{};
    // 需要复制传入数组的值
    std::vector<uint32_t> input_ids{};
    // 1. 先初始化KV Cache的数据
    constexpr int num_head = 12;
    constexpr int head_dim = 64;
    constexpr int max_seq = 16;
    constexpr int kv_manager_num = 512;
    constexpr int kv_mem_size = num_head * head_dim * sizeof(float16_t) * 2; // 每次存取的时候会存储K和V
    std::vector<unsigned int> kv_block_ids;
    KVCacheManager::initialize(kv_manager_num, kv_mem_size);
    const auto root_id = KVCacheManager::getInstance().root_id();
    // 获取KV Cache的主要的叶子root_id
    std::cout << "root_id = " << root_id << std::endl;
    // BIScheduler::set(BIScheduler::Type::OMP);
    BIScheduler::get().set_num_threads(std::thread::hardware_concurrency());
    BIMemoryGroup group{BIMemoryManagerOnDemand::make_default()};

    int batch_size = 1;
    int seq_len = 1;
    // 1. 初始化一个最大input算子
    BITensor original_input_tensor;
    BITensorShape original_input_tensor_shape(max_seq, 20);
    original_input_tensor.allocator()->init(BITensorInfo(original_input_tensor_shape, 1, BIDataType::U32));
    original_input_tensor.allocator()->allocate();

    // 1.1 初始化一个小型算子
    BITensor input_tensor;
    BITensorShape input_tensor_shape(seq_len, batch_size);
    BITensorInfo input_info(input_tensor_shape, 1, BIDataType::U32);
    input_info.set_format(Format::U32);
    input_tensor.allocator()->init(*original_input_tensor.allocator(), input_info);
    std::vector<uint32_t> indices_data{0, 1, 2};
    KVCacheTestName::fill_tensor_val_with_arr(input_tensor, indices_data);


    // 2. Gather的权重
    BITensorShape gather_weight_shape(768, 6003);
    const std::string &weight_path =
            "./input_res/transformer_wte_weight.npy";
    BITensor weight = utils::create_type_tensor(
        weight_path, gather_weight_shape,
        BIDataType::F16);

    // 3. 输出原始矩阵
    BITensor original_gather_output_tensor;
    BITensorShape original_gather_output_tensor_shape(768, max_seq, 20);
    BITensorShape original_attn_output_tensor_shape(768, 1, 20);
    original_gather_output_tensor.allocator()->init(
        BITensorInfo(original_gather_output_tensor_shape, 1, BIDataType::F16));
    original_gather_output_tensor.allocator()->allocate();
    BITensor original_attn_rms_output_tensor;
    original_attn_rms_output_tensor.allocator()->init(
        BITensorInfo(original_attn_output_tensor_shape, 1, BIDataType::F16));
    original_attn_rms_output_tensor.allocator()->allocate();
    // 3.1 输出矩阵的子矩阵
    BITensor gather_output_tensor;
    BITensorShape gather_output_tensor_shape(768, seq_len, batch_size);
    BITensorShape attn_output_tensor_shape(768, 1, batch_size);
    BITensorInfo gather_output_info(gather_output_tensor_shape, 1, BIDataType::F16);
    BITensorInfo attn_output_info(attn_output_tensor_shape, 1, BIDataType::F16);
    gather_output_info.set_format(Format::F16);
    attn_output_info.set_format(Format::F16);
    gather_output_tensor.allocator()->init(*original_gather_output_tensor.allocator(), gather_output_info);
    // 2. 进行NEGather筛选
    BINEGather gather_layer;
    gather_layer.configure(&weight, &input_tensor, &gather_output_tensor, 1);
    gather_layer.run();
    // 3. Add权重的获取
    BITensorShape add_wte_weight_shape(768, max_seq);
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
    BITensor original_split_output_tensor;
    original_split_output_tensor.allocator()->init(
        BITensorInfo(original_attn_output_tensor_shape, 1, BIDataType::F16));
    original_split_output_tensor.allocator()->allocate();
    BITensor split_add_output_tensor;
    split_add_output_tensor.allocator()->init(*original_split_output_tensor.allocator(), attn_output_info);
    BINEArithmeticAddition add_layer;
    add_layer.configure(&gather_output_tensor, &sub_add_weight, &add_output_tensor, BIConvertPolicy::SATURATE);
    add_layer.run();
    BIITensorPack pack;
    pack.add_tensor(ACL_SRC, &add_output_tensor);
    pack.add_tensor(ACL_DST, &split_add_output_tensor);
    BINEScheduler::get().schedule_kv_split(pack);
    // 5 获取Attention模块的权重
    // 5.1 gamma权重
    BITensorShape attn_gamma_weights_shape(768);
    const std::string &gamma_weights_path = "./input_res/attn_gamma_weights.npy";
    BITensor attn_gamma_weights = utils::create_type_tensor(
        gamma_weights_path, attn_gamma_weights_shape,
        BIDataType::F16);
    BITensor attn_origin_o_tensor;
    attn_origin_o_tensor.allocator()->init(BITensorInfo(original_attn_output_tensor_shape, 1, BIDataType::F16));
    attn_origin_o_tensor.allocator()->allocate();
    BITensor attn_output_tensor;
    attn_output_tensor.allocator()->init(*attn_origin_o_tensor.allocator(), attn_output_info);
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
    attn_layer.configure(&split_add_output_tensor,
                         &attn_gamma_weights,
                         &attn_qkv_weights,
                         &attn_qkv_bias,
                         &attn_c_proj_weights,
                         &attn_c_proj_bias, q_perm,
                         k_perm,
                         qkv_o_perm,
                         768,
                         max_seq,
                         20,
                         &attn_output_tensor);
    attn_layer.run();
    attn_layer.get_kv_block_ids(kv_block_ids);
    BITensor sub_mlp_input;
    sub_mlp_input.allocator()->init(*original_attn_rms_output_tensor.allocator(), attn_output_info);
    BINEArithmeticAddition attn_rms_add; // 注意力RMS相加
    attn_rms_add.configure(&split_add_output_tensor, &attn_output_tensor, &sub_mlp_input, BIConvertPolicy::SATURATE);
    attn_rms_add.run();
    BINEFeedForwardLayer _mlp_layer; // MLP层
    //2. 初始化gamma张量
    const std::string &gamma_path = "./input_res/mlp_rms_gamma.npy";
    BITensor gamma = KVCacheTestName::create_norm_input(std::vector<int>{768}, gamma_path);
    // 3. 初始化fc_weights的权重
    const std::string &c_fc_weights_path =
            "./input_res/reordered_c_fc_weights.npy";
    BITensor c_fc_weights = utils::create_type_tensor(c_fc_weights_path, BITensorShape(3072, 768), BIDataType::F16);
    // 4. 初始化fc_bias
    const std::string &c_fc_bias_path = "./input_res/c_fc_bias.npy";
    BITensor c_fc_bias = utils::create_type_tensor(c_fc_bias_path, BITensorShape(3072), BIDataType::F16);
    // 5. 输出张量
    BITensor output;
    output.allocator()->init(BITensorInfo(BITensorShape(768, 1, 20), 1, BIDataType::F16));
    output.allocator()->allocate();
    BITensor sub_mlp_output;
    sub_mlp_output.allocator()->init(*output.allocator(), attn_output_info);
    // 6. proj的权重
    const std::string &c_proj_path = "./input_res/c_proj_weights.npy";
    BITensor c_proj_weight = KVCacheTestName::create_norm_input(std::vector<int>{3072, 768}, c_proj_path);
    const std::string &c_proj_bias_path =
            "./input_res/c_proj_bias.npy";
    BITensor c_proj_bias = KVCacheTestName::create_norm_input(std::vector<int>{768}, c_proj_bias_path);
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
                         1);
    _mlp_layer.run(); // 1. 先用输出结果进行相加
    BITensorShape add_output_shape(768, 1, 20);
    BITensor add_output;
    add_output.allocator()->init(BITensorInfo(add_output_shape, 1, BIDataType::F16));
    add_output.allocator()->allocate();
    BITensor sub_add_output;
    sub_add_output.allocator()->init(*add_output.allocator(), attn_output_info);
    BINEArithmeticAddition add_f;
    add_f.configure(&sub_mlp_output, &sub_mlp_input, &sub_add_output, BIConvertPolicy::SATURATE);
    add_f.run();
    // 2. 对结果再进行一次归一化
    BITensor mlp_after_gamma = KVCacheTestName::create_norm_input(std::vector{768},
                                                                  "./input_res/mlp_after_rms_gamma.npy");
    BITensor mlp_rms_output;
    mlp_rms_output.allocator()->init(BITensorInfo(add_output_shape, 1, BIDataType::F16));
    mlp_rms_output.allocator()->allocate();
    BITensor sub_mlp_rms_output;
    sub_mlp_rms_output.allocator()->init(*mlp_rms_output.allocator(), attn_output_info);
    BINERMSNormLayer rms_norm_layer;
    rms_norm_layer.configure(&sub_add_output, &mlp_after_gamma, &sub_mlp_rms_output);
    rms_norm_layer.run();
    // 3. 对输出结果进行LMHead操作
    BITensor lm_head_weights = KVCacheTestName::create_norm_input(std::vector{768, 6003},
                                                                  "./input_res/lm_head_weights.npy");
    BITensor lm_head_output;
    lm_head_output.allocator()->init(BITensorInfo(BITensorShape(6003, 1, 20), 1, BIDataType::F16));
    lm_head_output.allocator()->allocate();
    BITensor sub_lm_head_output;
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
    lm_head_layer.configure(&sub_mlp_rms_output, &lm_head_weights, nullptr, &sub_lm_head_output, 1.0f, 1.0f, gemm_info);
    lm_head_layer.run();
    // // MemAllocTest::print_tensor(sub_mlp_output, "attn_output_tensor");
    BITensor ids;
    ids.allocator()->init(BITensorInfo(BITensorShape(1, 20), 1, BIDataType::S32));
    ids.allocator()->allocate();
    BITensor sub_ids;
    BITensorInfo sub_ids_info = BITensorInfo(BITensorShape(1, batch_size), 1, BIDataType::S32);
    sub_ids_info.set_format(Format::S32);
    sub_ids.allocator()->init(*ids.allocator(), sub_ids_info);

    BINEArgMinMaxLayer arg_minmax_layer;
    arg_minmax_layer.configure(&sub_lm_head_output, 0, &sub_ids, BIReductionOperation::ARG_IDX_MAX);
    arg_minmax_layer.run();
    std::vector<int> infos{};
    std::vector<float> score{};
    KVCacheTestName::concat_tensor(sub_ids, output_ids);
    KVCacheTestName::get_s32_val(sub_ids, infos);
    KVCacheTestName::get_index_val(sub_lm_head_output, infos, score);
    scores.push_back(score[0]);

    KVCacheTestName::print_output_info(output_ids);
    KVCacheTestName::print_output_info(scores);
    // 再次进行运行(动态)
    batch_size = 20;
    std::vector<std::vector<unsigned int> > inp_map{};
    for (int i = 0; i < batch_size; i++) {
        inp_map.push_back({kv_block_ids[0], 1});
    }
    seq_len = 2;
    input_tensor_shape = BITensorShape(seq_len, batch_size);
    input_info.set_tensor_shape(input_tensor_shape);
    input_tensor.allocator()->init(*original_input_tensor.allocator(), input_info);
    indices_data = {0, 3};
    KVCacheTestName::fill_tensor_with_repeat_arr(input_ids, batch_size, indices_data);
    KVCacheTestName::fill_tensor_val_with_arr(input_tensor, input_ids);
    gather_output_tensor_shape = BITensorShape(768, seq_len, batch_size);
    attn_output_tensor_shape = BITensorShape(768, 1, batch_size);
    gather_output_info.set_tensor_shape(gather_output_tensor_shape);
    attn_output_info.set_tensor_shape(attn_output_tensor_shape);
    gather_output_tensor.allocator()->init(*original_gather_output_tensor.allocator(), gather_output_info);
    add_output_tensor.allocator()->init(*original_add_output_tensor.allocator(), gather_output_info);
    split_add_output_tensor.allocator()->init(*original_split_output_tensor.allocator(), attn_output_info);
    sub_add_weight_shape = BITensorShape(768, seq_len);
    sub_add_weight_info.set_tensor_shape(sub_add_weight_shape);
    sub_add_weight.allocator()->init(*add_wte_weight.allocator(), sub_add_weight_info);
    attn_output_tensor.allocator()->init(*attn_origin_o_tensor.allocator(), attn_output_info);
    sub_mlp_input.allocator()->init(*original_attn_rms_output_tensor.allocator(), attn_output_info);
    sub_mlp_output.allocator()->init(*output.allocator(), attn_output_info);
    sub_mlp_rms_output.allocator()->init(*mlp_rms_output.allocator(), attn_output_info);
    sub_add_output.allocator()->init(*add_output.allocator(), attn_output_info);
    sub_lm_head_output_info.set_tensor_shape(BITensorShape(6003, 1, batch_size));
    sub_lm_head_output.allocator()->init(*lm_head_output.allocator(), sub_lm_head_output_info);
    sub_ids_info.set_tensor_shape(BITensorShape(1, batch_size));
    sub_ids.allocator()->init(*ids.allocator(), sub_ids_info);
    gather_layer.dynamic_configure(&input_tensor, &gather_output_tensor);
    add_layer.dynamic_configure(&gather_output_tensor, &sub_add_weight, true);
    attn_layer.dynamic_configure(&split_add_output_tensor, seq_len, batch_size, inp_map);
    attn_rms_add.dynamic_configure(&split_add_output_tensor, &attn_output_tensor, true);
    _mlp_layer.dynamic_configure(&sub_mlp_input, batch_size);
    add_f.dynamic_configure(&sub_mlp_output, &sub_mlp_input, false);
    rms_norm_layer.dynamic_configure(&sub_add_output);
    lm_head_layer.dynamic_configure();
    arg_minmax_layer.configure(&sub_lm_head_output, 0, &sub_ids, BIReductionOperation::ARG_IDX_MAX);
    gather_layer.run();
    add_layer.run();
    BINEScheduler::get().schedule_kv_split(pack);
    attn_layer.run();
    attn_layer.get_kv_block_ids(kv_block_ids);
    // attn_rms_add.run();
    // _mlp_layer.run();
    inp_map.clear();
    for (unsigned int &kv_block_id: kv_block_ids)
        inp_map.push_back({kv_block_id, 1});
    add_f.run();
    rms_norm_layer.run();
    lm_head_layer.run();
    arg_minmax_layer.run();
    KVCacheTestName::concat_tensor(sub_ids, output_ids);
    KVCacheTestName::get_s32_val(sub_ids, infos);
    KVCacheTestName::get_index_val(sub_lm_head_output, infos, score);
    scores.push_back(score[0]);

    KVCacheTestName::print_output_info(output_ids);
    KVCacheTestName::print_output_info(scores);

    seq_len = 3;
    input_tensor_shape = BITensorShape(seq_len, batch_size);
    input_info.set_tensor_shape(input_tensor_shape);
    input_tensor.allocator()->init(*original_input_tensor.allocator(), input_info);
    indices_data = {0, 3, 4};
    KVCacheTestName::fill_tensor_with_repeat_arr(input_ids, batch_size, indices_data);
    KVCacheTestName::fill_tensor_val_with_arr(input_tensor, input_ids);
    gather_output_tensor_shape = BITensorShape(768, seq_len, batch_size);
    attn_output_tensor_shape = BITensorShape(768, 1, batch_size);
    gather_output_info.set_tensor_shape(gather_output_tensor_shape);
    attn_output_info.set_tensor_shape(attn_output_tensor_shape);
    gather_output_tensor.allocator()->init(*original_gather_output_tensor.allocator(), gather_output_info);
    add_output_tensor.allocator()->init(*original_add_output_tensor.allocator(), gather_output_info);
    split_add_output_tensor.allocator()->init(*original_split_output_tensor.allocator(), attn_output_info);
    sub_add_weight_shape = BITensorShape(768, seq_len);
    sub_add_weight_info.set_tensor_shape(sub_add_weight_shape);
    sub_add_weight.allocator()->init(*add_wte_weight.allocator(), sub_add_weight_info);
    attn_output_tensor.allocator()->init(*attn_origin_o_tensor.allocator(), attn_output_info);
    sub_mlp_input.allocator()->init(*original_attn_rms_output_tensor.allocator(), attn_output_info);
    sub_mlp_output.allocator()->init(*output.allocator(), attn_output_info);
    sub_mlp_rms_output.allocator()->init(*mlp_rms_output.allocator(), attn_output_info);
    sub_add_output.allocator()->init(*add_output.allocator(), attn_output_info);
    sub_lm_head_output_info.set_tensor_shape(BITensorShape(6003, 1, batch_size));
    sub_lm_head_output.allocator()->init(*lm_head_output.allocator(), sub_lm_head_output_info);
    sub_ids_info.set_tensor_shape(BITensorShape(1, batch_size));
    sub_ids.allocator()->init(*ids.allocator(), sub_ids_info);
    gather_layer.dynamic_configure(&input_tensor, &gather_output_tensor);
    add_layer.dynamic_configure(&gather_output_tensor, &sub_add_weight, true);
    attn_layer.dynamic_configure(&split_add_output_tensor, seq_len, batch_size, inp_map);
    attn_rms_add.dynamic_configure(&split_add_output_tensor, &attn_output_tensor, true);
    _mlp_layer.dynamic_configure(&sub_mlp_input, batch_size);
    add_f.dynamic_configure(&sub_mlp_output, &sub_mlp_input, false);
    rms_norm_layer.dynamic_configure(&sub_add_output);
    lm_head_layer.dynamic_configure();
    arg_minmax_layer.configure(&sub_lm_head_output, 0, &sub_ids, BIReductionOperation::ARG_IDX_MAX);
    gather_layer.run();
    add_layer.run();
    BINEScheduler::get().schedule_kv_split(pack);
    attn_layer.run();
    attn_layer.get_kv_block_ids(kv_block_ids);
    attn_rms_add.run();
    _mlp_layer.run();
    inp_map.clear();
    for (unsigned int &kv_block_id: kv_block_ids)
        inp_map.push_back({kv_block_id, 1});
    add_f.run();
    rms_norm_layer.run();
    lm_head_layer.run();
    arg_minmax_layer.run();
    KVCacheTestName::concat_tensor(sub_ids, output_ids);
    KVCacheTestName::get_s32_val(sub_ids, infos);
    KVCacheTestName::get_index_val(sub_lm_head_output, infos, score);
    scores.push_back(score[0]);

    KVCacheTestName::print_output_info(output_ids);
    KVCacheTestName::print_output_info(scores);
    KVCacheTestName::print_tensor(sub_ids, "sub_ids");

    for (int seq_run = 1; seq_run < 13; seq_run++) {
        seq_len++;
        std::cout << "当前的sequence长度" << seq_len << std::endl;
        input_tensor_shape = BITensorShape(seq_len, batch_size);
        input_info.set_tensor_shape(input_tensor_shape);
        input_tensor.allocator()->init(*original_input_tensor.allocator(), input_info);
        indices_data.push_back(seq_run + 4);
        KVCacheTestName::fill_tensor_with_repeat_arr(input_ids, batch_size, indices_data);
        KVCacheTestName::fill_tensor_val_with_arr(input_tensor, input_ids);
        gather_output_tensor_shape = BITensorShape(768, seq_len, batch_size);
        attn_output_tensor_shape = BITensorShape(768, 1, batch_size);
        gather_output_info.set_tensor_shape(gather_output_tensor_shape);
        attn_output_info.set_tensor_shape(attn_output_tensor_shape);
        gather_output_tensor.allocator()->init(*original_gather_output_tensor.allocator(), gather_output_info);
        add_output_tensor.allocator()->init(*original_add_output_tensor.allocator(), gather_output_info);
        split_add_output_tensor.allocator()->init(*original_split_output_tensor.allocator(), attn_output_info);
        sub_add_weight_shape = BITensorShape(768, seq_len);
        sub_add_weight_info.set_tensor_shape(sub_add_weight_shape);
        sub_add_weight.allocator()->init(*add_wte_weight.allocator(), sub_add_weight_info);
        attn_output_tensor.allocator()->init(*attn_origin_o_tensor.allocator(), attn_output_info);
        sub_mlp_input.allocator()->init(*original_attn_rms_output_tensor.allocator(), attn_output_info);
        sub_mlp_output.allocator()->init(*output.allocator(), attn_output_info);
        sub_mlp_rms_output.allocator()->init(*mlp_rms_output.allocator(), attn_output_info);
        sub_add_output.allocator()->init(*add_output.allocator(), attn_output_info);
        sub_lm_head_output_info.set_tensor_shape(BITensorShape(6003, 1, batch_size));
        sub_lm_head_output.allocator()->init(*lm_head_output.allocator(), sub_lm_head_output_info);
        sub_ids_info.set_tensor_shape(BITensorShape(1, batch_size));
        sub_ids.allocator()->init(*ids.allocator(), sub_ids_info);
        auto start = std::chrono::high_resolution_clock::now();
        gather_layer.dynamic_configure(&input_tensor, &gather_output_tensor);
        add_layer.dynamic_configure(&gather_output_tensor, &sub_add_weight, true);
        attn_layer.dynamic_configure(&split_add_output_tensor, seq_len, batch_size, inp_map);
        attn_rms_add.dynamic_configure(&split_add_output_tensor, &attn_output_tensor, true);
        _mlp_layer.dynamic_configure(&sub_mlp_input, batch_size);
        add_f.dynamic_configure(&sub_mlp_output, &sub_mlp_input, false);
        rms_norm_layer.dynamic_configure(&sub_add_output);
        lm_head_layer.dynamic_configure();
        arg_minmax_layer.configure(&sub_lm_head_output, 0, &sub_ids, BIReductionOperation::ARG_IDX_MAX);
        gather_layer.run();
        add_layer.run();
        BINEScheduler::get().schedule_kv_split(pack);
        attn_layer.run();
        // KVCacheTestName::print_tensor(attn_output_tensor, "attn_output_tensor");
        attn_layer.get_kv_block_ids(kv_block_ids);
        attn_rms_add.run();
        _mlp_layer.run();
        inp_map.clear();
        for (unsigned int &kv_block_id: kv_block_ids)
            inp_map.push_back({kv_block_id, 1});
        KVCacheTestName::print_tensor(split_add_output_tensor, "scores");
        add_f.run();
        rms_norm_layer.run();
        lm_head_layer.run();
        auto end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double, std::milli>(end - start).count();
        arg_minmax_layer.run();
        std::cout << "Performance Report:" << duration << std::endl;
        KVCacheTestName::print_tensor(sub_ids, "sub_ids");
        // KVCacheTestName::print_tensor(sub_lm_head_output, "scores");
        KVCacheTestName::concat_tensor(sub_ids, output_ids);
        KVCacheTestName::get_s32_val(sub_ids, infos);
        KVCacheTestName::get_index_val(sub_lm_head_output, infos, score);
        scores.push_back(score[0]);

        KVCacheTestName::print_output_info(output_ids);
        KVCacheTestName::print_output_info(scores);
        // KVCacheTestName::print_tensor(sub_lm_head_output, "scores");
    }
}

TEST(KVCacheGPT, TensorPartAlloc) {
    // 1. 先初始化Tensor
    constexpr int batch_size = 20;
    constexpr int seq_len = 3;
    constexpr int hidden_size = 768;
    BITensorShape input_tensor_shape = BITensorShape(hidden_size, seq_len, batch_size);
    BITensor input_tensor, kv_tensor;
    BITensorInfo input_t_info = BITensorInfo(input_tensor_shape, 1, BIDataType::F16);
    std::cout << input_t_info.dimension(3) << std::endl;
    kv_tensor.allocator()->init(BITensorInfo(BITensorShape(hidden_size, 1, batch_size), 1, BIDataType::F16));
    kv_tensor.allocator()->allocate();
    input_tensor.allocator()->init(input_t_info);
    input_tensor.allocator()->allocate();
    KVCacheTestName::fill_tensor_val_with_index<float16_t>(input_tensor);
    KVCacheTestName::print_tensor(input_tensor, "src");
    BIITensorPack pack;
    pack.add_tensor(ACL_SRC, &input_tensor);
    pack.add_tensor(ACL_DST, &kv_tensor);
    BatmanInfer::BINEScheduler::get().schedule_kv_split(pack);
    KVCacheTestName::print_tensor(kv_tensor, "dst");
}

TEST(KVCacheGPT, TensorConcatAlloc) {
    // // 1. 先初始化Tensor
    // constexpr int batch_size = 14;
    // constexpr int seq_len = 3;
    // constexpr int num_head = 4;
    // constexpr int head_dim = 16;
    // BITensor src_t1, src_t2, dst_t1, dst_t2;
    // src_t1.allocator()->init(BITensorInfo(BITensorShape(head_dim, num_head, 1, batch_size), 1, BIDataType::F16));
    // src_t1.allocator()->allocate();
    // src_t2.allocator()->init(BITensorInfo(BITensorShape(head_dim, num_head, 1, batch_size), 1, BIDataType::F16));
    // src_t2.allocator()->allocate();
    // dst_t1.allocator()->init(BITensorInfo(BITensorShape(head_dim, num_head, seq_len, batch_size), 1, BIDataType::F16));
    // dst_t1.allocator()->allocate();
    // dst_t2.allocator()->init(BITensorInfo(BITensorShape(head_dim, num_head, seq_len, batch_size), 1, BIDataType::F16));
    // dst_t2.allocator()->allocate();
    // // 2. 先获取中间两个tensor的信息
    // std::vector<BITensor> k_cache_tensors{};
    // std::vector<BITensor> v_cache_tensors{};
    // for (int i = 0; i < batch_size * (seq_len - 1); i++) {
    //     BITensor k_tmp_tensor;
    //     BITensor v_tmp_tensor;
    //     k_tmp_tensor.allocator()->init(BITensorInfo(BITensorShape(head_dim, num_head), 1,
    //                                                 BIDataType::F16));
    //     v_tmp_tensor.allocator()->init(BITensorInfo(BITensorShape(head_dim, num_head), 1,
    //                                                 BIDataType::F16));
    //     k_tmp_tensor.allocator()->allocate();
    //     v_tmp_tensor.allocator()->allocate();
    //     KVCacheTestName::fill_tensor_val_with_index<float16_t>(k_tmp_tensor);
    //     KVCacheTestName::fill_tensor_val_with_index<float16_t>(v_tmp_tensor);
    //     k_cache_tensors.push_back(std::move(k_tmp_tensor));
    //     v_cache_tensors.push_back(std::move(v_tmp_tensor));
    // }
    // KVCacheTestName::fill_tensor_val_with_index<float16_t>(src_t1);
    // KVCacheTestName::fill_tensor_val_with_index<float16_t>(src_t2);
    //
    // // 3.生成两个Physical Block
    // // 3.1 先预开辟一个大的内存
    // constexpr int block_size = sizeof(float16_t) * head_dim * num_head * 2;
    // auto memory_pool = aligned_alloc(64, 2 * batch_size * block_size);
    // std::vector<PhysicalBlock *> blocks{};
    // for (int i = 0; i < 2 * batch_size; i++) {
    //     auto block_buffer_ptr = static_cast<char *>(memory_pool) + (i * block_size);
    //     auto *new_block = new PhysicalBlock{i, block_buffer_ptr};
    //     blocks.emplace_back(new_block);
    //     memcpy(blocks[i]->buffer, k_cache_tensors.at(i).buffer(), block_size / 2);
    //     memcpy(static_cast<char *>(blocks[i]->buffer) + block_size / 2, k_cache_tensors.at(i).buffer(), block_size / 2);
    // }
    //
    // BIITensorPack pack;
    // pack.add_tensor(ACL_SRC_0, &src_t1);
    // pack.add_tensor(ACL_SRC_1, &src_t2);
    // pack.add_tensor(ACL_DST_0, &dst_t1);
    // pack.add_tensor(ACL_DST_1, &dst_t2);
    // BINEScheduler::get().schedule_kv_concat(pack, blocks);
    // for (PhysicalBlock *block_ptr: blocks) {
    //     delete block_ptr;
    // }
    // blocks.clear(); // Clear the vector of pointers
    // KVCacheTestName::print_tensor(dst_t1, "dst");
}

TEST(KVCacheGPT, TensorGEMV) {
    BITensor src_tensor;
    BITensor src_tensor_2;
    constexpr int batch_size = 14;
    constexpr int seq_len = 3;
    constexpr int num_head = 4;
    constexpr int head_dim = 16;
    src_tensor.allocator()->init(BITensorInfo(BITensorShape(head_dim, 1, num_head, 1), 1, BIDataType::F16));
    src_tensor.allocator()->allocate();
    src_tensor_2.allocator()->init(BITensorInfo(BITensorShape(head_dim, 1, num_head, 1), 1, BIDataType::F16));
    src_tensor_2.allocator()->allocate();
    KVCacheTestName::fill_tensor_val_with_index<float16_t>(src_tensor);
    KVCacheTestName::fill_tensor_val_with_index_2<float16_t>(src_tensor_2);
    // 初始化一个Block的存储状态
    KVCacheManager::initialize(512, num_head * head_dim * sizeof(float16_t));
    auto root_id = KVCacheManager::getInstance().root_id();
    // 塞入一个buffer
    KVCacheManager::getInstance().memcpy_decode_buffer(src_tensor.buffer(), root_id);
    std::cout << root_id << std::endl;
    KVCacheManager::getInstance().alloc_decode_next(root_id, 1, std::vector<unsigned int>{3});
    auto decode_leafs = KVCacheManager::getInstance().get_decode_ids();
    for (auto decode_leaf: decode_leafs) {
        std::cout << "Block Id: " << decode_leaf.first << "\t";
        std::cout << "Decode Id: " << decode_leaf.second << std::endl;
        KVCacheManager::getInstance().memcpy_decode_buffer(src_tensor_2.buffer(),
                                                           decode_leaf.first);
        KVCacheTestName::print_offset<float16_t>(KVCacheManager::getInstance().decode_buffer_ptr(decode_leaf.first),
                                                 num_head * head_dim);
    }
    KVCacheTestName::print_offset<float16_t>(KVCacheManager::getInstance().decode_buffer_ptr(root_id),
                                             num_head * head_dim);
}
