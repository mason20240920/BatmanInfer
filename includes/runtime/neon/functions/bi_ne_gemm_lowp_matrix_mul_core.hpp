//
// Created by Mason on 2025/1/3.
//

#ifndef BATMANINFER_BI_NE_GEMM_LOWP_MATRIX_MUL_CORE_HPP
#define BATMANINFER_BI_NE_GEMM_LOWP_MATRIX_MUL_CORE_HPP

#include <runtime/bi_i_function.hpp>
#include <runtime/bi_i_memory_manager.hpp>
#include <runtime/bi_i_weights_manager.hpp>
#include <runtime/bi_memory_manager_on_demand.hpp>
#include <data/core/bi_i_tensor.hpp>
#include <function_info/bi_GEMMInfo.h>

#include <memory>

namespace BatmanInfer {
    class BIITensor;

    class BIITensorInfo;

    class BINEGEMMLowpMatrixMultipleCore : public BIIFunction {
    public:
        BINEGEMMLowpMatrixMultipleCore(std::shared_ptr<BIIMemoryManager> memory_manager,
                                       BIIWeightsManager *weights_manager = nullptr);

        BINEGEMMLowpMatrixMultipleCore() : BINEGEMMLowpMatrixMultipleCore(BIMemoryManagerOnDemand::make_default()) {
        }

        BINEGEMMLowpMatrixMultipleCore(const BINEGEMMLowpMatrixMultipleCore &) = delete;

        BINEGEMMLowpMatrixMultipleCore(BINEGEMMLowpMatrixMultipleCore &&) = default;

        BINEGEMMLowpMatrixMultipleCore &operator=(const BINEGEMMLowpMatrixMultipleCore &) = delete;

        BINEGEMMLowpMatrixMultipleCore &operator=(BINEGEMMLowpMatrixMultipleCore &&) = default;

        ~BINEGEMMLowpMatrixMultipleCore();

        /**
         * @brief 初始化内核的输入和输出
         *
         *        支持的数据布局:
         *            NHWC
         *
         *        Valid data type configurations:
         * |src0           |src1               |src2     |dst            |
         * |:--------------|:------------------|:--------|:--------------|
         * |QASYMM8        |QASYMM8            |S32      |QASYMM8        |
         * |QASYMM8        |QASYMM8_SIGNED     |S32      |QASYMM8        |
         * |QASYMM8        |QSYMM8_PER_CHANNEL |S32      |QASYMM8        |
         * |QASYMM8        |QSYMM8             |S32      |QASYMM8        |
         * |QASYMM8        |QASYMM8            |S32      |S32            |
         * |QASYMM8        |QSYMM8_PER_CHANNEL |S32      |S32            |
         * |QASYMM8        |QSYMM8             |S32      |S32            |
         * |QASYMM8        |QASYMM8_SIGNED     |F32      |F32            |
         * |QASYMM8_SIGNED |QASYMM8_SIGNED     |S32      |QASYMM8_SIGNED |
         * |QASYMM8_SIGNED |QSYMM8_PER_CHANNEL |S32      |QASYMM8_SIGNED |
         * |QASYMM8_SIGNED |QSYMM8             |S32      |QASYMM8_SIGNED |
         * |QASYMM8_SIGNED |QASYMM8_SIGNED     |S32      |S32            |
         * |QASYMM8_SIGNED |QSYMM8_PER_CHANNEL |S32      |S32            |
         * |QASYMM8_SIGNED |QSYMM8             |S32      |S32            |
         * |QASYMM8_SIGNED |QASYMM8_SIGNED     |F32      |F32            |
         *
         * @note GEMM_LOWP 是一个低精度 GEMM 内核。
         *       1. 将矩阵 A 的 QASYMM8 值转换为 int32，并为每个值加上偏移量 a_offset。 (避免溢出)
         *       2. 将矩阵 B 的 QASYMM8 值转换为 int32，并为每个值加上偏移量 b_offset
         *       3. 计算转换后的矩阵 A 和矩阵 B 的乘积，结果为 int32 类型
         * @param a 第一个输入张量（矩阵 A）。支持的数据类型：QASYMM8/QASYMM8_SIGNED
         * @param b 第二个输入张量（矩阵 B）。支持的数据类型：QASYMM8/QASYMM8_SIGNED/QSYMM8/QSYMM8_PER_CHANNEL
         * @param c 可以为空。支持的数据类型：S32/F32
         * @param output 输出张量。支持的数据类型：S32/QASYMM8/QASYMM8_SIGNED/F32
         * @param gemm_info 可选参数，指定矩阵 A 和/或矩阵 B 是否已经被重排，以及矩阵 B 的重排是否仅在首次运行时执行
         */
        void configure(const BIITensor *a,
                       const BIITensor *b,
                       const BIITensor *c,
                       BIITensor *output,
                       const GEMMInfo &gemm_info = GEMMInfo());

        void dynamic_configure(const BIITensor *input, const BIITensor *output) const;

        /**
         * @brief 静态函数检查给定的信息是否会导致有效的配置 @ref NEGEMMLowpMatrixMultiplyCore。
         * @param a
         * @param b
         * @param c
         * @param output
         * @param gemm_info
         * @return 返回状态
         */
        static BIStatus validate(const BIITensorInfo *a,
                                 const BIITensorInfo *b,
                                 const BIITensorInfo *c,
                                 const BIITensorInfo *output,
                                 const GEMMInfo &gemm_info = GEMMInfo());

        /**
         * @brief 在运行阶段更新量化信息，以便能够正确计算量化乘数。
         *        请查看 NEGEMMConvolutionLayer.h 以获取更深入的解释和示例。
         */
        void update_quantization_parameters();

        void run();

        void prepare() override;

    private:
        struct Impl;
        std::unique_ptr<Impl> _impl;
    };
}

#endif //BATMANINFER_BI_NE_GEMM_LOWP_MATRIX_MUL_CORE_HPP
