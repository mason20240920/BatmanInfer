//
// Created by Mason on 2025/1/2.
//

#ifndef BATMANINFER_BI_NE_GEMM_HPP
#define BATMANINFER_BI_NE_GEMM_HPP

#include <function_info/bi_GEMMInfo.h>
#include <runtime/bi_i_function.hpp>
#include <runtime/bi_i_memory_manager.hpp>
#include <runtime/bi_i_weights_manager.hpp>
#include <runtime/bi_memory_manager_on_demand.hpp>

namespace BatmanInfer {
    /** 基本函数用于执行 GEMM。此函数调用以下内核：
     *
     * -# cpu::CpuGemm
     * */
    class BINEGEMM : public BIIFunction {
    public:
        BINEGEMM(std::shared_ptr<BIIMemoryManager>
                 memory_manager,
                 BIIWeightsManager *weights_manager = nullptr
        );

        BINEGEMM() : BINEGEMM(BIMemoryManagerOnDemand::make_default()) {
        }

        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BINEGEMM(const BINEGEMM &) = delete;

        /** Default move constructor */
        BINEGEMM(BINEGEMM
            &&) = default;

        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BINEGEMM &operator=(const BINEGEMM &) = delete;

        /** Default move assignment operator */
        BINEGEMM &operator=(BINEGEMM &&) = default;

        /** Default destructor */
        ~BINEGEMM();

        /**
         * 初始化内核的输入和输出
          *
          * 支持的数据布局：
          * - 全部
          *
          * 支持的数据类型配置：
          * |src0         |src1        |src2      |dst            |
          * |:------------|:-----------|:---------|:--------------|
          * |F32          |F32         |F32       |F32            |
          * |F16          |F16         |F16       |F16            |
          * |BFLOAT16     |BFLOAT16    |BFLOAT16  |BFLOAT16       |
          *
          * @note GEMM: 通用矩阵乘法 - [alpha * A * B + beta * C]。
          * @note GEMM: 张量 a、b、c 和 d 必须具有相同的数据类型。在调用此函数时，不应混用不同的数据类型。
          *
          * @note 批量 GEMM 仅支持右侧矩阵 (RHS) 的维度(rank) 小于左侧矩阵 (LHS) 的广播情况，不支持反过来的情况。
          *
          * @param[in]  a         第一个输入张量（矩阵 A 或向量 A）。支持的数据类型：BFLOAT16/F16/F32
          * @param[in]  b         第二个输入张量（矩阵 B）。支持的数据类型：与 @p a 相同
          * @param[in]  c         第三个输入张量（矩阵 C）。如果只需要计算 @p a 和 @p b 的乘积，可以为 nullptr。支持的数据类型：与 @p a 相同
          * @param[out] d         输出张量。支持的数据类型：与 @p a 相同
          * @param[in]  alpha     矩阵乘积的权重
          * @param[in]  beta      矩阵 C 的权重
          * @param[in]  gemm_info (可选) 指定矩阵 A 和/或矩阵 B 是否已被重排，
          *                       以及矩阵 B 的重排是否仅在第一次运行时进行
         */
        void configure(const BIITensor *a,
                       const BIITensor *b,
                       const BIITensor *c,
                       BIITensor *d,
                       float alpha,
                       float beta,
                       const GEMMInfo &gemm_info = GEMMInfo());

        void dynamic_configure() const;

        /** Static function to check if given info will lead to a valid configuration of @ref NEGEMM.
         *
         * Similar to @ref NEGEMM::configure()
         *
         * @return a status
         */
        static BIStatus validate(const BIITensorInfo *a,
                                 const BIITensorInfo *b,
                                 const BIITensorInfo *c,
                                 const BIITensorInfo *output,
                                 float alpha,
                                 float beta,
                                 const GEMMInfo &gemm_info = GEMMInfo());

        /** Static function that queries whether there exists fixed-format kernel and if it exists it will return in the first argument in what format
         * weights are expected to be reshaped as defined by WeightFormat class. Apart from the first argument the rest of the arguments are the same
        * as in @ref NEGEMM::validate() except that all arguments are required.
        *
        * @return a status
        */
        static BIStatus has_opt_impl(BatmanInfer::BIWeightFormat &expected_weight_format,
                                     const BIITensorInfo *a,
                                     const BIITensorInfo *b,
                                     const BIITensorInfo *c,
                                     const BIITensorInfo *output,
                                     float alpha,
                                     float beta,
                                     const GEMMInfo &gemm_info = GEMMInfo());

        // Inherited methods overridden:
        void run();

        void prepare() override;

    private:
        struct Impl;
        std::unique_ptr<Impl> _impl;
    };
}

#endif //BATMANINFER_BI_NE_GEMM_HPP
