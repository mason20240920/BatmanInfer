//
// Created by Mason on 2025/4/10.
//
#include <thread>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <runtime/neon/bi_ne_functions.h>
#include <runtime/bi_tensor.hpp>

#include "runtime/bi_scheduler.hpp"
#include "utils/utils.hpp"

namespace MemAllocTest {
    void print_tensor(const BatmanInfer::BITensor &tensor,
                      const std::string &name = "temp",
                      const BatmanInfer::BIIOFormatInfo::PrintRegion region =
                              BatmanInfer::BIIOFormatInfo::PrintRegion::Full) {
        std::cout << name << std::endl;
        BatmanInfer::BIIOFormatInfo format;
        format.element_delim = ", "; // 元素之间用逗号分隔
        format.row_delim = "\n"; // 每行换行
        format.align_columns = 1; // 对齐列
        format.print_region = region;

        tensor.print(std::cout, format);
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
    attn_output_tensor.allocator()->init(*original_add_output_tensor.allocator(), gather_output_info);

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

    // 再次进行运行(动态)
    batch_size = 2;
    seq_len = 4;
    input_tensor_shape = BITensorShape(seq_len, batch_size);
    input_info.set_tensor_shape(input_tensor_shape);
    input_tensor.allocator()->init(*original_input_tensor.allocator(), input_info);
    indices_data = {0, 8, 9, 10, 0, 8, 9, 10};
    MemAllocTest::fill_tensor_val_with_arr(input_tensor, indices_data);
    gather_output_tensor_shape = BITensorShape(768, seq_len, batch_size);
    gather_output_info.set_tensor_shape(gather_output_tensor_shape);
    gather_output_tensor.allocator()->init(*original_gather_output_tensor.allocator(), gather_output_info);
    add_output_tensor.allocator()->init(*original_add_output_tensor.allocator(), gather_output_info);
    sub_add_weight_shape = BITensorShape(768, seq_len);
    sub_add_weight_info.set_tensor_shape(sub_add_weight_shape);
    sub_add_weight.allocator()->init(*add_wte_weight.allocator(), sub_add_weight_info);
    attn_output_tensor.allocator()->init(*original_add_output_tensor.allocator(), gather_output_info);

    gather_layer.dynamic_configure(&input_tensor, &gather_output_tensor);
    add_layer.dynamic_configure(&gather_output_tensor, &sub_add_weight, true);
    attn_lowp_layer.dynamic_configure(&add_output_tensor, seq_len, batch_size);

    std::cout << "====================" << std::endl;

    gather_layer.run();
    add_layer.run();
    attn_lowp_layer.run();


    // const auto warmup = 10; // 预热次数
    // const auto iterations = 1000; // 运行次数
    // const double outlier_threshold = 3.0; // 异常值阈值(标准差倍数)
    // std::vector<double> timings;
    // timings.reserve(iterations);
    //
    // // 预测阶段（不记录时间）
    // for (size_t i = 0; i < warmup; ++i) {
    //     attn_lowp_layer.dynamic_configure(&add_output_tensor, seq_len, batch_size);
    //     attn_lowp_layer.run(); // 预测阶段（不记录时间）
    // } // 正式测量
    // for (size_t i = 0; i < iterations; ++i) {
    //     auto start = std::chrono::high_resolution_clock::now();
    //     attn_lowp_layer.dynamic_configure(&add_output_tensor, seq_len, batch_size);
    //     attn_lowp_layer.run(); // 预测阶段（不记录时间）
    //     auto end = std::chrono::high_resolution_clock::now();
    //
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
    //
    // // 应用3-sigma法则过滤异常值
    // std::vector<double> filtered;
    // std::copy_if(timings.begin(), timings.end(), std::back_inserter(filtered),
    //              [=](double x) { return std::abs(x - avg) < outlier_threshold * std_dev; });
    //
    // // 重新计算统计量
    // double valid_avg = std::accumulate(filtered.begin(), filtered.end(), 0.0) / filtered.size();
    // auto [min_it, max_it] = std::minmax_element(filtered.begin(), filtered.end());
    //
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
