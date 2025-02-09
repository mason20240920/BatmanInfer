//
// Created by Mason on 2025/2/9.
//

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <neon/neon_defines.h>
#include <runtime/neon/bi_ne_functions.h>
#include "runtime/bi_scheduler.hpp"

/**
 * 使用Neon指令加速计算
 * @param input
 * @param scale
 * @param bias
 */
void layer_norm_optimized(float *output,     // 输出指针 [batch_size, seq_len, hidden_size]
                          const float *input,// 输入指针 [batch_size, seq_len, hidden_size]
                          const float *scale,// 可训练参数 [hidden_size]
                          const float *bias, // 可训练参数 [hidden_size]
                          int batch_size,
                          int seq_len,
                          int hidden_size,
                          float epsilon = 1e-5f) {
    const int vec_size = 4; // NEON向量宽度(float32x4_t)
    const int num_blocks = hidden_size / vec_size; // 隐藏层进行整除

    for (int b = 0; b < batch_size; ++b) {
        for (int s = 0; s < seq_len; ++s) {
            // 当前token的起始位置
            const float *x = input + b * seq_len * hidden_size + s * hidden_size;
            float *out = output + b * seq_len * hidden_size + s * hidden_size;

            // 步骤1: 计算均值(向量化累累加)
            float sum = 0.0f;
            for (int i = 0; i < num_blocks; ++i) {
                float32x4_t vec = vld1q_f32(x + i * vec_size);
                sum += vaddvq_f32(vec);  // 向量内4元素求和
            }
            float mu = sum / hidden_size;

            // 步骤2：计算标准差（向量化平方差）
            float var_sum = 0.0f;
            float32x4_t mu_vec = vdupq_n_f32(mu);
            for (int i = 0; i < num_blocks; ++i) {
                float32x4_t vec = vld1q_f32(x + i * vec_size);
                float32x4_t diff = vsubq_f32(vec, mu_vec);
                var_sum += vaddvq_f32(vmulq_f32(diff, diff)); // 平方差求和
            }
            float sigma = sqrtf(var_sum / hidden_size + epsilon);

            // 步骤3：标准化+仿射变换
            float32x4_t sigma_vec = vdupq_n_f32(sigma);
            for (int i = 0; i < num_blocks; ++i) {
                float32x4_t vec = vld1q_f32(x + i * vec_size);
                float32x4_t diff = vsubq_f32(vec, mu_vec);
                float32x4_t normalized = vdivq_f32(diff, sigma_vec);

                // 加载当前block对应的scale和bias
                float32x4_t scale_vec = vld1q_f32(scale + i * vec_size);
                float32x4_t bias_vec = vld1q_f32(bias + i * vec_size);

                // 计算: normalized * scale + bias
                float32x4_t result = vmlaq_f32(bias_vec, normalized, scale_vec);

                // 存储结果
                vst1q_f32(out + i * vec_size, result);
            }
        }
    }
}

/**
 * 内存对齐分配器（16字节对齐）
 * @param size
 * @return
 */
float *aligned_alloc_float(size_t size) {
    // NEON要求的最小对齐
    const size_t alignment = 16;
#ifdef BI_COMPUTE_LOGGING_ENABLED
    void *ptr = aligned_alloc(alignment, size * sizeof(float));
#else
    void *ptr = nullptr;
#endif
    if (!ptr) {
        perror("Memory allocation failed");
        exit(EXIT_FAILURE);
    }
    return static_cast<float *>(ptr);
}

void init_params(float *scale,
                 float *bias,
                 int size) {
    for (int i = 0; i < size; ++i) {
        scale[i] = 1.0f;
        bias[i] = 0.0f;
    }
}

void free_aligned(float *ptr) {
    free(ptr);
}

TEST(NEONOperator, LayerNormalizer) {
    const int batch_size = 32;
    const int seq_len = 128;
    const int hidden_size = 768;  // 必须为4的倍数

    // 分配输入输出内存（自动对齐）
    float *input = aligned_alloc_float(batch_size * seq_len * hidden_size);
    float *output = aligned_alloc_float(batch_size * seq_len * hidden_size);

    // 初始化输入数据（示例：随机值）
    std::memset(input, 1, batch_size * seq_len * hidden_size * sizeof(float)); // 实际应填充真实数据

    // 初始化参数
    float scale[hidden_size], bias[hidden_size];
    init_params(scale, bias, hidden_size);

    auto start = std::chrono::high_resolution_clock::now();

    // 执行LayerNorm
    layer_norm_optimized(output, input, scale, bias,
                         batch_size, seq_len, hidden_size);

    auto end = std::chrono::high_resolution_clock::now();

    // 释放内存
    free_aligned(input);
    free_aligned(output);

    // 计算时间差
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // 打印结果
    std::cout << "Function execution time: " << duration.count() << " microseconds" << std::endl;
}

TEST(NEONOperator, NENormalizationLayer) {
    using namespace BatmanInfer;

    const int batch_size = 32;
    const int seq_len = 128;
    const int hidden_size = 768;  // 必须满足ACL对齐要求

    // 1. 配置Tensor信息 (NHWC格式)
    BITensorShape input_shape(batch_size, hidden_size, seq_len);
    BITensorInfo input_info(input_shape, 1, BIDataType::F16);
    BITensorInfo output_info(input_shape, 1, BIDataType::F16);

    // 2. 创建Tensor对象
    BITensor input, output;
    input.allocator()->init(input_info);
    output.allocator()->init(output_info);

    // 3. 分配内存
    input.allocator()->allocate();
    output.allocator()->allocate();

    // 4. 初始化参数 (使用ACL规范参数)
    const BINormalizationLayerInfo norm_info(
            BINormType::CROSS_MAP, // 归一化类型
            5,                                // 归一化窗口大小
            0.0001f,                          // epsilon
            0.75f,                            // beta
            1.0f,                             // kappa
            false                             // 是否跨通道
    );

    // 5. 创建并配置归一化层
    BINENormalizationLayer norm_layer;
    norm_layer.configure(&input, &output, norm_info);

    // 6. 填充输入数据 (示例填充1.0)
    float *input_data = reinterpret_cast<float *>(input.buffer());
    std::fill_n(input_data, batch_size * seq_len * hidden_size, 1.0f);

    auto start = std::chrono::high_resolution_clock::now();

    // 7. 执行归一化
    norm_layer.run();

    auto end = std::chrono::high_resolution_clock::now();

    // 8. 释放内存
    input.allocator()->free();
    output.allocator()->free();

    // 时间计算保持不变
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "ACL execution time: " << duration.count() << " microseconds" << std::endl;
}