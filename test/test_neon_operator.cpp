//
// Created by Mason on 2025/2/9.
//

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <neon/neon_defines.h>
#include <runtime/neon/bi_ne_functions.h>
#include "runtime/bi_scheduler.hpp"
#include <cpu/kernels/layer_norm/generic/neon/fp16.hpp>
#include <utils/utils.hpp>

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
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // 输出运行时间
    std::cout << "Current Sequence Length: " << seq_len << std::endl;
    std::cout << "Function execution time: " << duration.count() << " microseconds" << std::endl;

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

/**
 * 假设输入输出为对齐的16字节内存地址
 * RMSNorm 高性能实现 (公式: y = γ * x / sqrt(mean(x²) + ε))
 * @param output 结果数组 [out] 对齐的16字节输出数组
 * @param input 输入数组 [in]  对齐的16字节输入数组
 * @param gamma 缩放系数
 * @param epsilon 1e-5
 * @param num_elements 元素总数(需要为8的倍数)
 */
void rms_norm_neon_fp16(float16_t *output,
                        const float16_t *input,
                        const float16_t *gamma,
                        const float16_t epsilon,
                        int num_elements) {
    // 公式: sum_sq = Σ(x_i^2)

    // 阶段1: 平方和计算 (优化指令级并行) --------------------------------
    float16x8_t sum_sq_v0 = vdupq_n_f16(0.0f); // 使用双累加器消除数据依赖
    float16x8_t sum_sq_v1 = vdupq_n_f16(0.0f);
    int i = 0;

    // 循环展开4次 (32 elements/iteration), 减少循环开销
    for (; i <= num_elements - 64; i += 64) {
        // 预取下个缓存行（ARM典型缓存行64字节）
        __builtin_prefetch(input + i + 64);

        // 加载8个向量（64字节）
        float16x8_t x0 = vld1q_f16(input + i);
        float16x8_t x1 = vld1q_f16(input + i + 8);
        float16x8_t x2 = vld1q_f16(input + i + 16);
        float16x8_t x3 = vld1q_f16(input + i + 24);
        float16x8_t x4 = vld1q_f16(input + i + 32);
        float16x8_t x5 = vld1q_f16(input + i + 40);
        float16x8_t x6 = vld1q_f16(input + i + 48);
        float16x8_t x7 = vld1q_f16(input + i + 56);

        // 交错计算以隐藏指令延迟
        sum_sq_v0 = vfmaq_f16(sum_sq_v0, x0, x0);
        sum_sq_v1 = vfmaq_f16(sum_sq_v1, x1, x1);
        sum_sq_v0 = vfmaq_f16(sum_sq_v0, x2, x2);
        sum_sq_v1 = vfmaq_f16(sum_sq_v1, x3, x3);
        sum_sq_v0 = vfmaq_f16(sum_sq_v0, x4, x4);
        sum_sq_v1 = vfmaq_f16(sum_sq_v1, x5, x5);
        sum_sq_v0 = vfmaq_f16(sum_sq_v0, x6, x6);
        sum_sq_v1 = vfmaq_f16(sum_sq_v1, x7, x7);
    }

    // 合并累加器
    sum_sq_v0 = vaddq_f16(sum_sq_v0, sum_sq_v1);

    // 处理剩余元素（使用单累加器）
    for (; i <= num_elements - 8; i += 8) {
        float16x8_t x = vld1q_f16(input + i);
        sum_sq_v0 = vfmaq_f16(sum_sq_v0, x, x);
    }

    // 阶段2: 归约计算 (保持高精度) -------------------------------------
    float32x4_t sum_low = vcvt_f32_f16(vget_low_f16(sum_sq_v0));
    float32x4_t sum_high = vcvt_f32_f16(vget_high_f16(sum_sq_v0));
    sum_low = vaddq_f32(sum_low, sum_high);

    float32x2_t sum_half = vadd_f32(vget_low_f32(sum_low), vget_high_f32(sum_low));
    float sum_sq = vget_lane_f32(vpadd_f32(sum_half, sum_half), 0);

    // 阶段3: 计算缩放因子 ---------------------------------------------
    const auto N = static_cast<float >(num_elements);
    const auto eps_f32 = (float) epsilon; // 正确使用标量转换函数
    const float rms_inv = 1.0f / sqrtf(sum_sq / N + eps_f32);
    const float16_t rms_inv_f16 = vduph_lane_f16(vcvt_f16_f32(vdupq_n_f32(rms_inv)), 0);
    const float16x8_t rms_inv_v = vdupq_n_f16(rms_inv_f16); // 向量化的rms_inv


    // 阶段4: 应用缩放 (优化存储指令) -----------------------------------
    i = 0;
    for (; i <= num_elements - 32; i += 32) {
        // 加载input和gamma的4个向量
        float16x8_t x0 = vld1q_f16(input + i);
        float16x8_t g0 = vld1q_f16(gamma + i);
        float16x8_t x1 = vld1q_f16(input + i + 8);
        float16x8_t g1 = vld1q_f16(gamma + i + 8);
        float16x8_t x2 = vld1q_f16(input + i + 16);
        float16x8_t g2 = vld1q_f16(gamma + i + 16);
        float16x8_t x3 = vld1q_f16(input + i + 24);
        float16x8_t g3 = vld1q_f16(gamma + i + 24);

        // 计算：output = (x * gamma) * rms_inv
        vst1q_f16(output + i, vmulq_f16(vmulq_f16(x0, g0), rms_inv_v));
        vst1q_f16(output + i + 8, vmulq_f16(vmulq_f16(x1, g1), rms_inv_v));
        vst1q_f16(output + i + 16, vmulq_f16(vmulq_f16(x2, g2), rms_inv_v));
        vst1q_f16(output + i + 24, vmulq_f16(vmulq_f16(x3, g3), rms_inv_v));
    }

    // 处理尾部元素
    for (; i <= num_elements - 8; i += 8) {
        float16x8_t x = vld1q_f16(input + i);
        float16x8_t g = vld1q_f16(gamma + i);
        vst1q_f16(output + i, vmulq_f16(vmulq_f16(x, g), rms_inv_v));
    }
}

using namespace BatmanInfer;

// 替代vaddvq_f16的手动归约实现
inline float16_t manual_addv_f16(float16x8_t vec) {
    // 步骤分解：将128位寄存器分解为两个64位部分
    float16x4_t low = vget_low_f16(vec);
    float16x4_t high = vget_high_f16(vec);

    // 横向归约：((a+b)+(c+d)) + ((e+f)+(g+h))
    float16x4_t sum1 = vpadd_f16(low, high);      // [a+b, c+d, e+f, g+h]
    float16x4_t sum2 = vpadd_f16(sum1, sum1);     // [(a+b)+(c+d), (e+f)+(g+h), ...]
    return vget_lane_f16(sum2, 0) + vget_lane_f16(sum2, 1);
}

void rms_norm_acl_tensor_fp16(BatmanInfer::BITensor &output,
                              const BatmanInfer::BITensor &input,
                              const BatmanInfer::BITensor &gamma,
                              float epsilon = 1e-5f) {
    // 条件检查
    BI_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&input, 1, BIDataType::F16);
    BI_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(&input, &gamma, &output);
    // 检查 H 维度匹配
    BI_COMPUTE_ERROR_ON(input.info()->dimension(0) != gamma.info()->dimension(0));
    // 检查是否能被8整除
    BI_COMPUTE_ERROR_ON(gamma.info()->dimension(0) % 8 != 0);

    // 获取 Tensor 信息
    const auto in_shape = input.info()->tensor_shape();
    const int H = static_cast<int>(in_shape[0]); // hidden_size （最内核维度）
    const int S = static_cast<int>(in_shape[1]); // sequence_length （序列长度)
    const int step = 8;

    // 配置执行窗口
    BIWindow window;
    window.set(BIWindow::DimX, BIWindow::BIDimension(0, 1)); // H维度固定
    window.set(BIWindow::DimY, BIWindow::BIDimension(0, S)); // 遍历S维度

    BIIterator input_it(&input, window);
    BIIterator output_it(&output, window);

    // 阶段1: 平方和计算 (优化指令级并行) --------------------------------

    // 双累加器初始化（每个窗口迭代独立）
    float16x8_t sum_sq_v0 = vdupq_n_f16(0.0f);
    float16x8_t sum_sq_v1 = vdupq_n_f16(0.0f);

    // 平方和计算阶段
    execute_window_loop(window, [&](const BICoordinates &id) {
        //  获取当前处理位置的指针
        const auto *in_ptr = reinterpret_cast<const float16_t *>(input_it.ptr());

        // 窗口步长8时，直接展开4次处理32个元素（无需内部循环）
        // 假设window的x维度步长是8，这里处理4个连续的8元素块
        {
            // 预取策略：提前预取下一个窗口的数据
            __builtin_prefetch(in_ptr + 64); // 预取下一个缓存行

            // 加载8个向量（64字节）
            float16x8_t x0 = vld1q_f16(in_ptr);
            float16x8_t x1 = vld1q_f16(in_ptr + 8);
            float16x8_t x2 = vld1q_f16(in_ptr + 16);
            float16x8_t x3 = vld1q_f16(in_ptr + 24);
            float16x8_t x4 = vld1q_f16(in_ptr + 32);
            float16x8_t x5 = vld1q_f16(in_ptr + 40);
            float16x8_t x6 = vld1q_f16(in_ptr + 48);
            float16x8_t x7 = vld1q_f16(in_ptr + 56);

            // 交错计算以隐藏指令延迟
            sum_sq_v0 = vfmaq_f16(sum_sq_v0, x0, x0);
            sum_sq_v1 = vfmaq_f16(sum_sq_v1, x1, x1);
            sum_sq_v0 = vfmaq_f16(sum_sq_v0, x2, x2);
            sum_sq_v1 = vfmaq_f16(sum_sq_v1, x3, x3);
            sum_sq_v0 = vfmaq_f16(sum_sq_v0, x4, x4);
            sum_sq_v1 = vfmaq_f16(sum_sq_v1, x5, x5);
            sum_sq_v0 = vfmaq_f16(sum_sq_v0, x6, x6);
            sum_sq_v1 = vfmaq_f16(sum_sq_v1, x7, x7);
        }


    }, input_it);

    // 合并累加器
    sum_sq_v0 = vaddq_f16(sum_sq_v0, sum_sq_v1);

    // 阶段2: 归约计算 (保持高精度) -------------------------------------
    float32x4_t sum_low = vcvt_f32_f16(vget_low_f16(sum_sq_v0));
    float32x4_t sum_high = vcvt_f32_f16(vget_high_f16(sum_sq_v0));
    sum_low = vaddq_f32(sum_low, sum_high);
    float32x2_t sum_half = vadd_f32(vget_low_f32(sum_low), vget_high_f32(sum_low));
    float sum_sq = vget_lane_f32(vpadd_f32(sum_half, sum_half), 0);

    // 阶段3: 计算缩放因子 ---------------------------------------------
}

void run_kernel(const BIITensor *input,
                const BIITensor *gamma,
                BIITensor *output,
                const float16_t epsilon = 1e-5) {
    // 创建窗口
    BIWindow win;
    win.set(BIWindow::DimX, BIWindow::BIDimension(0, 1));
    win.set(BIWindow::DimY, BIWindow::BIDimension(0, input->info()->dimension(1)));

    BIIterator input_it(input, win);
    BIIterator output_it(output, win);

    execute_window_loop(win, [&](const BICoordinates &id) {
        // 获取输入/输出的数据指针
        auto in_ptr = reinterpret_cast<const float16_t *>(input_it.ptr());
        auto out_ptr = reinterpret_cast<float16_t *>(output_it.ptr());

        // 阶段1: 平方和计算 (优化指令级并行) --------------------------------

        // 双累加器初始化（每个窗口迭代独立）
        float32x4_t sum_sq_v0 = vdupq_n_f32(0.0f);
        float32x4_t sum_sq_v1 = vdupq_n_f32(0.0f);
        int i = 0;
        const int N = input->info()->dimension(0);

        // 循环展开4次 (32 elements/iteration), 减少循环开销
        for (; i <= N - 64; i += 64) {
            // 预取下个缓存行（ARM典型缓存行64字节）
            __builtin_prefetch(input + i + 64);

            // 加载8个向量（64字节）
            float16x8_t x0 = vld1q_f16(in_ptr + i);
            float16x8_t x1 = vld1q_f16(in_ptr + i + 8);
            float16x8_t x2 = vld1q_f16(in_ptr + i + 16);
            float16x8_t x3 = vld1q_f16(in_ptr + i + 24);
            float16x8_t x4 = vld1q_f16(in_ptr + i + 32);
            float16x8_t x5 = vld1q_f16(in_ptr + i + 40);
            float16x8_t x6 = vld1q_f16(in_ptr + i + 48);
            float16x8_t x7 = vld1q_f16(in_ptr + i + 56);

            // 交错计算以隐藏指令延迟
            // 转换为 float32，并计算平方累加
            sum_sq_v0 = vfmaq_f32(sum_sq_v0, vcvt_f32_f16(vget_low_f16(x0)), vcvt_f32_f16(vget_low_f16(x0)));
            sum_sq_v0 = vfmaq_f32(sum_sq_v0, vcvt_f32_f16(vget_high_f16(x0)), vcvt_f32_f16(vget_high_f16(x0)));

            sum_sq_v1 = vfmaq_f32(sum_sq_v1, vcvt_f32_f16(vget_low_f16(x1)), vcvt_f32_f16(vget_low_f16(x1)));
            sum_sq_v1 = vfmaq_f32(sum_sq_v1, vcvt_f32_f16(vget_high_f16(x1)), vcvt_f32_f16(vget_high_f16(x1)));

            sum_sq_v0 = vfmaq_f32(sum_sq_v0, vcvt_f32_f16(vget_low_f16(x2)), vcvt_f32_f16(vget_low_f16(x2)));
            sum_sq_v0 = vfmaq_f32(sum_sq_v0, vcvt_f32_f16(vget_high_f16(x2)), vcvt_f32_f16(vget_high_f16(x2)));

            sum_sq_v1 = vfmaq_f32(sum_sq_v1, vcvt_f32_f16(vget_low_f16(x3)), vcvt_f32_f16(vget_low_f16(x3)));
            sum_sq_v1 = vfmaq_f32(sum_sq_v1, vcvt_f32_f16(vget_high_f16(x3)), vcvt_f32_f16(vget_high_f16(x3)));

            sum_sq_v0 = vfmaq_f32(sum_sq_v0, vcvt_f32_f16(vget_low_f16(x4)), vcvt_f32_f16(vget_low_f16(x4)));
            sum_sq_v0 = vfmaq_f32(sum_sq_v0, vcvt_f32_f16(vget_high_f16(x4)), vcvt_f32_f16(vget_high_f16(x4)));

            sum_sq_v1 = vfmaq_f32(sum_sq_v1, vcvt_f32_f16(vget_low_f16(x5)), vcvt_f32_f16(vget_low_f16(x5)));
            sum_sq_v1 = vfmaq_f32(sum_sq_v1, vcvt_f32_f16(vget_high_f16(x5)), vcvt_f32_f16(vget_high_f16(x5)));

            sum_sq_v0 = vfmaq_f32(sum_sq_v0, vcvt_f32_f16(vget_low_f16(x6)), vcvt_f32_f16(vget_low_f16(x6)));
            sum_sq_v0 = vfmaq_f32(sum_sq_v0, vcvt_f32_f16(vget_high_f16(x6)), vcvt_f32_f16(vget_high_f16(x6)));

            sum_sq_v1 = vfmaq_f32(sum_sq_v1, vcvt_f32_f16(vget_low_f16(x7)), vcvt_f32_f16(vget_low_f16(x7)));
            sum_sq_v1 = vfmaq_f32(sum_sq_v1, vcvt_f32_f16(vget_high_f16(x7)), vcvt_f32_f16(vget_high_f16(x7)));
        }

        // 合并累加器
        sum_sq_v0 = vaddq_f32(sum_sq_v0, sum_sq_v1);

        // 阶段2: 归约计算 (保持高精度) -------------------------------------
        float32x2_t sum_half = vadd_f32(vget_low_f32(sum_sq_v0), vget_high_f32(sum_sq_v0));
        float sum_sq = vget_lane_f32(vpadd_f32(sum_half, sum_half), 0);


        // 阶段3: 计算缩放因子 ---------------------------------------------
        const auto eps_f32 = (float) epsilon; // 正确使用标量转换函数
        const float rms_inv = 1.0f / sqrtf(sum_sq / N + eps_f32);

        const float16_t rms_inv_f16 = vduph_lane_f16(vcvt_f16_f32(vdupq_n_f32(rms_inv)), 0);
        const float16x8_t rms_inv_v = vdupq_n_f16(rms_inv_f16); // 向量化的rms_inv



        auto gamma_ptr = reinterpret_cast<const float16_t *>(gamma->buffer());
        i = 0;
        for (; i <= N - 32; i += 32) {
            // 加载input和gamma的4个向量
            float16x8_t x0 = vld1q_f16(in_ptr + i);
            float16x8_t g0 = vld1q_f16(gamma_ptr + i);
            float16x8_t x1 = vld1q_f16(in_ptr + i + 8);
            float16x8_t g1 = vld1q_f16(gamma_ptr + i + 8);
            float16x8_t x2 = vld1q_f16(in_ptr + i + 16);
            float16x8_t g2 = vld1q_f16(gamma_ptr + i + 16);
            float16x8_t x3 = vld1q_f16(in_ptr + i + 24);
            float16x8_t g3 = vld1q_f16(gamma_ptr + i + 24);

            // 计算：output = (x * gamma) * rms_inv
            vst1q_f16(out_ptr + i, vmulq_f16(vmulq_f16(x0, g0), rms_inv_v));
            vst1q_f16(out_ptr + i + 8, vmulq_f16(vmulq_f16(x1, g1), rms_inv_v));
            vst1q_f16(out_ptr + i + 16, vmulq_f16(vmulq_f16(x2, g2), rms_inv_v));
            vst1q_f16(out_ptr + i + 24, vmulq_f16(vmulq_f16(x3, g3), rms_inv_v));
        }

    }, input_it, output_it);
}

TEST(ARMWindow, WindowTest) {
    // 输入格式
    BIIOFormatInfo format;
    format.element_delim = ", ";  // 元素之间用逗号分隔
    format.row_delim = "\n";      // 每行换行
    format.align_columns = 1;     // 对齐列
    BITensor input, output, gamma;

    const int N = 768;

    auto input_shape = BITensorShape(N, 16);
    auto output_shape = BITensorShape(N, 16);
    auto gamma_shape = BITensorShape(N);
    input.allocator()->init(BITensorInfo(input_shape, 1, BIDataType::F16));
    output.allocator()->init(BITensorInfo(output_shape, 1, BIDataType::F16));
    gamma.allocator()->init(BITensorInfo(gamma_shape, 1, BIDataType::F16));

    input.allocator()->allocate();
    output.allocator()->allocate();
    gamma.allocator()->allocate();

    std::vector<float16_t> input_data(N * 12), gamma_data(N);
    // 初始化输入数据（模拟正态分布）
    for (int i = 0; i < (N * 12); ++i)
        input_data[i] = static_cast<float16_t>((i % 32 - 16.0f) / 8.0f);
    for (int i = 0; i < N; ++i)
        gamma_data[i] = static_cast<float16_t>(1);

    copy_data_to_tensor(input, 0, input_data);
    copy_data_to_tensor(gamma, 0, gamma_data);

    // 开始时间节点
    auto start = std::chrono::high_resolution_clock::now();

    run_kernel(&input, &gamma, &output);

    // 结束时间节点
    auto end = std::chrono::high_resolution_clock::now();

    // 计算耗时（以微秒为单位）
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Function execution time: " << duration.count() << " microseconds" << std::endl;

//    input.print(std::cout, format);
//    output.print(std::cout, format);
}

TEST(NEONOperator, RMSNormLayerTest) {
    // 输入格式
    BIIOFormatInfo format;
    format.element_delim = ", ";  // 元素之间用逗号分隔
    format.row_delim = "\n";      // 每行换行
    format.align_columns = 1;     // 对齐列
    BITensor input, output, gamma;
    auto input_shape = BITensorShape(768, 1);
    auto gamma_shape = BITensorShape(768);

    const int N = 768 * 1;
    const int H = 768;

    input.allocator()->init(BITensorInfo(input_shape, 1, BIDataType::F16));
    output.allocator()->init(BITensorInfo(input_shape, 1, BIDataType::F16));
    gamma.allocator()->init(BITensorInfo(gamma_shape, 1, BIDataType::F16));

    input.allocator()->allocate();
    output.allocator()->allocate();
    gamma.allocator()->allocate();

    std::vector<float16_t> input_data(N), gamma_data(H);

    // 初始化输入数据（模拟正态分布）
    for (int i = 0; i < N; ++i) {
//        input_data[i] = static_cast<float16_t>(i);
        input_data[i] = static_cast<float16_t>((i % 32 - 16.0f) / 8.0f);
    }
    for (int i = 0; i < H; ++i) {
        gamma_data[i] = static_cast<float16_t>(1);
    }

    copy_data_to_tensor(input, 0, input_data);
    copy_data_to_tensor(gamma, 0, gamma_data);

    input.print(std::cout, format);
    gamma.print(std::cout, format);

    rms_norm_acl_tensor_fp16(output,
                             input,
                             gamma);

    output.print(std::cout, format);
}

TEST(NEONOperator, RMSNormNeonCode) {
    const int N = 768;
    float16_t input[N], output[N], gamma[N];
    const float16_t epsilon = 1e-5f;

    // 初始化输入数据（模拟正态分布）
    for (int i = 0; i < N; ++i) {
        input[i] = static_cast<float16_t>((i % 32 - 16.0f) / 8.0f);
        gamma[i] = static_cast<float16_t>(2);
    }

    // 开始时间节点
    auto start = std::chrono::high_resolution_clock::now();

    // 调用NEON优化函数
    rms_norm_neon_fp16(output, input, gamma, epsilon, N);
    // 结束时间节点
    auto end = std::chrono::high_resolution_clock::now();

    // 计算耗时（以微秒为单位）
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Function execution time: " << duration.count() << " microseconds" << std::endl;

    // 打印前8个结果
    std::cout << "NEON Output (first 8 elements):\n";
    for (int i = 0; i < 8; ++i) {
        std::cout << static_cast<float>(output[i]) << " ";
    }
    std::cout << "\n";
}

TEST(NEONOperator, LayerNorm) {
    using namespace BatmanInfer;

    // -------------------- 1. Tensor 配置 --------------------
    constexpr int hidden_size = 8;
    constexpr int seq_length = 4;

    // 创建输入输出张量 [H=768, S=16]
    BITensor input, output, gamma, beta;
    BITensorInfo input_info({hidden_size, seq_length}, 1, BIDataType::F16);
    BITensorInfo param_info({hidden_size}, 1, BIDataType::F16);

    input.allocator()->init(input_info);
    output.allocator()->init(input_info); // 输出与输入同维度
    gamma.allocator()->init(param_info);
    beta.allocator()->init(param_info);

    // 分配内存
    input.allocator()->allocate();
    output.allocator()->allocate();
    gamma.allocator()->allocate();
    beta.allocator()->allocate();

    // 填充测试数据（列优先）
    const std::vector<float16_t> src_data = {
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12,
            13, 14, 15, 16,
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12,
            13, 14, 15, 16,
    };

    const std::vector<float16_t> gamma_data = {
            1, 1, 1, 1, 1, 1, 1, 1
    };

    copy_data_to_tensor(input, 0, src_data);
    copy_data_to_tensor(gamma, 0, gamma_data);

    // -------------------- 3. 窗口配置 --------------------
    BIWindow window;
    window.use_tensor_dimensions(input.info()->tensor_shape());

    // 配置并行维度
    // Dim0: 隐藏维度（自动展开）
    // Dim1: 序列维度（每个窗口对应一个序列位置）
    window.set(0, BIWindow::BIDimension(0, hidden_size));
    window.set(1, BIWindow::BIDimension(0, seq_length));

    cpu::neon_layer_norm_float16_8_0_2D(window,
                                        &input,    // ACL Tensor 会自动转换为 ITensor 接口
                                        &gamma,
                                        &beta,
                                        &output);


    // 输入格式
    BIIOFormatInfo format;
    format.element_delim = ", ";  // 元素之间用逗号分隔
    format.row_delim = "\n";      // 每行换行
    format.align_columns = 1;     // 对齐列

    output.print(std::cout, format);
}

BITensor create_tensor(const std::string &file_name,
                       const BITensorShape &shape) {
    BITensor tensor;
    BITensorInfo tensor_info(shape, 1, BIDataType::F16);
    tensor.allocator()->init(tensor_info);
    tensor.allocator()->allocate();
    utils::read_npy_to_tensor(file_name, tensor);

    return tensor;
}

BITensor create_normal_tensor(const BITensorShape &shape) {
    BITensor tensor;
    BITensorInfo tensor_info(shape, 1, BIDataType::F16);
    tensor.allocator()->init(tensor_info);
    tensor.allocator()->allocate();

    return tensor;
}

void print_npy_tensor(const BITensor &tensor) {
    // 输入格式
    BIIOFormatInfo format;
    format.element_delim = ", ";  // 元素之间用逗号分隔
    format.row_delim = "\n";      // 每行换行
    format.align_columns = 1;     // 对齐列
    tensor.print(std::cout, format);
}

TEST(NEONOperator, TensorReader) {
    // 输入张量
    BITensorShape input_shape(768, 16, 1);
    auto input = create_normal_tensor(input_shape);
    auto gemm_output = create_normal_tensor(input_shape);
    auto gemm_output_1 = create_normal_tensor(BITensorShape(2304, 16));
    std::vector<float16_t> values(768 * 16);
    for (int i = 0; i < 768 * 16; ++i)
        values[i] = static_cast<float16_t>(static_cast<float>(i) / 10000.0f);
    memcpy(input.buffer(), values.data(), 768 * 16 * sizeof(float16_t));

//    print_npy_tensor(input);

    // RMS NORM参数
    BITensorShape rms_norm_shape(768);
    auto rms_norm_1 = create_tensor("/Users/mason/Downloads/gpt2_create/rms_attention_1.npy", rms_norm_shape);

    // Gemm操作
    BITensorShape weights_shape(2304, 768);
    auto weights = create_tensor("/Users/mason/Downloads/gpt2_create/attn_c_attn_weight.npy", weights_shape);

    BITensorShape bias_shape(2304);
    auto bias = create_tensor("/Users/mason/Downloads/gpt2_create/attn_c_attn_bias.npy", bias_shape);

    // 设置Split输出的Tensor模块
    BITensorShape split_shape = BITensorShape(768, 16, 1);
    auto split_0 = create_normal_tensor(split_shape);
    auto split_1 = create_normal_tensor(split_shape);
    auto split_2 = create_normal_tensor(split_shape);
    std::vector<BIITensor *> splits_out{&split_0, &split_1, &split_2};

    BINERMSNormLayer norm_layer;
    norm_layer.configure(&input, &rms_norm_1, &gemm_output);

    GEMMInfo gemm_info;
    gemm_info.set_fast_math(true);

    BINEGEMM gemm_layer;
    gemm_layer.configure(&gemm_output, &weights, &bias, &gemm_output_1, 1.0f, 1.0f, gemm_info);

    norm_layer.run();
    gemm_layer.run();

    print_npy_tensor(gemm_output_1);
}