//
// Created by Mason on 2025/2/20.
//

#include <benchmark/benchmark.h>
#include <runtime/bi_tensor.hpp>
#include <runtime/neon/bi_ne_functions.h>
#include <utils/utils.hpp>
#include <runtime/bi_scheduler.hpp>
#include <thread>
#include <omp.h>

using namespace BatmanInfer;

template<typename T>
void fill_perf_tensor_val(const BITensor &tensor, const T val) {
    auto tensor_ptr = reinterpret_cast<T *>(tensor.buffer());
    size_t num_elements = tensor.info()->tensor_shape().total_size(); // 获取元素数量
    for (size_t i = 0; i < num_elements; ++i) {
        tensor_ptr[i] = val;
    }
}

static void BM_RunCoreLogic(benchmark::State &state) {
//    BIScheduler::set(BIScheduler::Type::OMP);
    BIScheduler::get().set_num_threads(std::thread::hardware_concurrency());
    // 先确定需要的算子
    BINEAttentionLayer attention_layer;
    BINEArithmeticAddition add_f;
    BINEFeedForwardLayer feedforward_layer;
    BINEArithmeticAddition add_2_f;

    // 输入张量
    const BITensorShape input_shape(768,  // hidden size
                                    16,  // sequence length
                                    5);  // batch size
    const BITensorShape gamma_shape(768);
    const BITensorShape fc_weights_shape(3072,     // input_size (width, 匹配input宽度)
                                         768);    // hidden_units (height)
    const BITensorShape fc_bias_shape(3072);    // hidden_units
    // 权重张量
    const BITensorShape proj_weights_shape2(768,     // input_size (width, 匹配input宽度)
                                            3072);    // hidden_units (height)
    const BITensorShape proj_bias_shape2(768);    // hidden_units

    const BITensorShape output_shape(768,    // hidden_units (width)
                                     16,
                                     5);     // batch_size (height)
    const BIActivationLayerInfo act_info(BIActivationFunction::GELU);

    // 权重张量
    const BITensorShape weights_shape(2304,     // input_size (width, 匹配input宽度)
                                      768);    // hidden_units (height)

    // 偏置矩阵
    const BITensorShape bias_shape(2304);    // hidden_units

    // 权重张量
    const BITensorShape weights_shape2(768,     // input_size (width, 匹配input宽度)
                                       768);    // hidden_units (height)

    // 偏置矩阵
    const BITensorShape bias_shape2(768);    // hidden_units

    // 标量
    const BITensorShape scalar_shape(1);

    // 相加权重
    const BITensorShape add_shape(16, 16);

    PermutationVector perm{0, 2, 1, 3};
    PermutationVector perm2{2, 0, 1, 3};
    PermutationVector perm_final{0, 2, 1, 3};

    auto input = utils::create_tensor(input_shape);
    const auto gamma = utils::create_npy_tensor("./input_res/rms_attention_1.npy", gamma_shape);
    const auto fc_weights = utils::create_npy_tensor("./input_res/mlp_c_fc_weight.npy",
                                                     fc_weights_shape);
    const auto fc_bias = utils::create_npy_tensor("./input_res/mlp_c_fc_bias.npy", fc_bias_shape);
    const auto proj_weights = utils::create_npy_tensor("./input_res/mlp_c_proj_weight.npy",
                                                       proj_weights_shape2);
    const auto proj_bias = utils::create_npy_tensor("./input_res/mlp_c_proj_bias.npy",
                                                    proj_bias_shape2);
    auto output = utils::create_tensor(output_shape);
    const auto weights = utils::create_npy_tensor("./input_res/attn_c_attn_weight.npy",
                                                  weights_shape);
    const auto bias = utils::create_npy_tensor("./input_res/attn_c_attn_bias.npy", bias_shape);
    const auto weights2 = utils::create_npy_tensor("./input_res/attn_c_proj_weight_2.npy",
                                                   weights_shape2);
    const auto bias2 = utils::create_npy_tensor("./input_res/attn_c_proj_bias_2.npy", bias_shape2);
    const auto gamma2 = utils::create_npy_tensor("./input_res/mlp_ln_2_weight.npy", gamma_shape);
    const auto scalar = utils::create_tensor(scalar_shape);
    const auto add_tensor = utils::create_npy_tensor("./input_res/_attn_Where_output_0.npy", add_shape);

    // 加法结果
    auto add_temp_out = utils::create_tensor(input_shape);
    auto ffn_out = utils::create_tensor(input_shape);
    auto final_out = utils::create_tensor(input_shape);

    fill_perf_tensor_val(scalar, static_cast<float16_t>(0.3535533845424652));


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

    const auto warmup = 10;  // 预热次数
    const auto iterations = 1000; // 运行次数
    const double outlier_threshold = 3.0; // 异常值阈值(标准差倍数)

    std::vector<double> timings;
    timings.reserve(iterations);

    for (auto _: state) {
        // 排除准备时间
        state.PauseTiming();
        state.ResumeTiming();
        attention_layer.run();
        add_f.run();
        feedforward_layer.run();
        add_2_f.run();
        state.PauseTiming();
        state.ResumeTiming();
    }
}

BENCHMARK(BM_RunCoreLogic)->MinTime(10.0) // 最小总运行时间
        ->Repetitions(5)  // 重复5组测试取平均值
        ->MeasureProcessCPUTime(); // 测量进程CPU时间

BENCHMARK_MAIN();
