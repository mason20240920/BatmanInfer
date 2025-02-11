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

TEST(NEONOperator, NEFeedForwardLayer) {
    using namespace BatmanInfer;
    // 手动定义输入数据
    const float16_t input_data[4] = {1.0f, -2.0f, 0.5f, 3.0f}; // 行优先存储

    // 2. 初始化Tensor
    BITensor src, dst;
    src.allocator()->init(BITensorInfo(BITensorShape(2U, 2U), 1, BIDataType::F16));
    dst.allocator()->init(BITensorInfo(BITensorShape(2U, 2U), 1, BIDataType::F16));
    src.allocator()->allocate();
    dst.allocator()->allocate();

    // 3. 手动拷贝数据到Tensor
    auto *src_ptr = reinterpret_cast<float16_t *>(src.buffer());
    std::memcpy(src_ptr, input_data, 4 * sizeof(float16_t));

    // 4. 配置GELU层
    BINEActivationLayer gelu;
    gelu.configure(&src, &dst, BIActivationLayerInfo(BIActivationLayerInfo::ActivationFunction::GELU));

    // 5. 运行计算
    gelu.run();

    // 6. 输出结果
    std::cout << "GELU Output:\n";

    BIIOFormatInfo format;
    format.element_delim = ", ";  // 元素之间用逗号分隔
    format.row_delim = "\n";      // 每行换行
    format.align_columns = 1;     // 对齐列

    dst.print(std::cout, format);
}

template<typename T>
void copy_data_to_tensor(BatmanInfer::BITensor &input, const int start_width, const std::vector<T> &vec) {
    auto *src_ptr = reinterpret_cast<T *>(input.buffer());
    std::memcpy(src_ptr + start_width * vec.size() * sizeof(uint8_t), vec.data(), vec.size() * sizeof(float16_t));
}

template<typename T>
void run_dynamic_gemm(BatmanInfer::BINEGEMM &gemm,
                      int seq_len,
                      BatmanInfer::BITensor &src,
                      BatmanInfer::BITensor &dst,
                      const std::vector<T> &origin_vec) {
    using namespace BatmanInfer;
    auto new_shape = BITensorShape(2048, seq_len);
    src.allocator()->info().set_tensor_shape(new_shape);
    dst.allocator()->info().set_tensor_shape(new_shape);
    copy_data_to_tensor(src, (seq_len - 1), origin_vec);
    // 输入格式
    BIIOFormatInfo format;
    format.element_delim = ", ";  // 元素之间用逗号分隔
    format.row_delim = "\n";      // 每行换行
    format.align_columns = 1;     // 对齐列
//    src.print(std::cout, format);

    // 开始时间节点
    auto start = std::chrono::high_resolution_clock::now();
    gemm.run();
    // 结束时间节点
    auto end = std::chrono::high_resolution_clock::now();

    // 计算耗时（以微秒为单位）
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // 输出运行时间
    std::cout << "Current Sequence Length" << seq_len << std::endl;
    std::cout << "Function execution time: " << duration.count() << " milliseconds" << std::endl;

//    dst.print(std::cout, format);
}

TEST(NEONOperator, NEGemmActLayer) {
    using namespace BatmanInfer;

    // 张量定义
    BITensor src, weights, bias, dst;
    const BITensorShape src_shape(2048, 4);  // W=3, H=2 （列优先）
    const BITensorShape weight_shape(2048, 2048); // 3x2矩阵转置为2x3
    const BITensorShape bias_shape(2048);

    // 初始化张量（内存布局重要！）
    src.allocator()->init(BITensorInfo(src_shape, 1, BIDataType::F16));
    weights.allocator()->init(BITensorInfo(weight_shape, 1, BIDataType::F16));
    bias.allocator()->init(BITensorInfo(bias_shape, 1, BIDataType::F16));
    dst.allocator()->init(BITensorInfo(BITensorShape(2048, 4), 1, BIDataType::F16));

    src.allocator()->allocate();
    weights.allocator()->allocate();
    bias.allocator()->allocate();
    dst.allocator()->allocate();

    // 配置融合GELU的GEMM
    BINEGEMM gemm;
    GEMMInfo gemm_info;
    gemm_info.set_fast_math(true);
    gemm_info.set_activation_info(BIActivationLayerInfo(
            BIActivationLayerInfo::ActivationFunction::GELU,
            0.044715f, 0.79788458f
    ));

    gemm.configure(&src, &weights, &bias, &dst, 1.0f, 1.0f, gemm_info);

    src.allocator()->info().set_tensor_shape(BITensorShape(2048, 1));
    dst.allocator()->info().set_tensor_shape(BITensorShape(2048, 1));


    // 填充测试数据（列优先）
    const std::vector<float16_t> src_data = {
            1, 2, 3, 4,
    };
    const std::vector<float16_t> weights_data = {
            0.5, 1.5, 1, 1,
            1.0, 2.0, 1, 1,
            1.5, 0.5, 1, 1,
            0.5, 0.5, 1, 1
    };

    const std::vector<float16_t> bias_data = {0.2, -0.1};


    copy_data_to_tensor(weights, 0, weights_data); // 实际是3x2的转置
    copy_data_to_tensor(bias, 0, bias_data);

    run_dynamic_gemm(gemm, 1, src, dst, src_data);
    const std::vector<float16_t> src_data2 = {
            2, 4, 6, 8,
    };
    run_dynamic_gemm(gemm, 2, src, dst, src_data2);
    const std::vector<float16_t> src_data3 = {
            3, 6, 9, 12,
    };
    run_dynamic_gemm(gemm, 3, src, dst, src_data3);
    run_dynamic_gemm(gemm, 4, src, dst, src_data3);

//
//    // 7. 释放内存（如果需要）
//    src.allocator()->free();
//    weights.allocator()->free();
//    bias.allocator()->free();
//    dst.allocator()->free();
}