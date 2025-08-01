//
// Created by Mason on 2025/1/15.
//

#pragma once

#include <data/core/bi_types.hpp>
#include <function_info/bi_activationLayerInfo.h>
#include <runtime/bi_i_function.hpp>
#include <runtime/bi_memory_manager_on_demand.hpp>

#include <memory>

namespace BatmanInfer {
    /**
     * 设置CPU矩乘实现:
     */
    class BICpuMatMulSettings {
    public:
        /**
         * 最快数学计算标识符
         * @return
         */
        bool fast_math() const {
            return _fast_math;
        }

        bool fixed_format() const {
            return _fixed_format;
        }

        // 设置标识符
        BICpuMatMulSettings &fast_math(bool f_math) {
            _fast_math = f_math;
            return *this;
        }

        BICpuMatMulSettings &fixed_format(bool fixed_format) {
            _fixed_format = fixed_format;
            return *this;
        }

    private:
        bool _fast_math{false};
        bool _fixed_format{false};
    };

    // 前向声明
    class BIITensor;

    class BIITensorInfo;

    class BIMatMulInfo;

    class BIStatus;


    class BINEMatMul : public BIIFunction {
    public:
        BINEMatMul(std::shared_ptr<BIIMemoryManager> memory_manager);

        BINEMatMul() : BINEMatMul(BIMemoryManagerOnDemand::make_default()) {
        }

        ~BINEMatMul();

        BINEMatMul(const BINEMatMul &) = delete;

        BINEMatMul(BINEMatMul &&) = default;

        BINEMatMul &operator=(const BINEMatMul &) = delete;

        BINEMatMul &operator=(BINEMatMul &&) = default;

        /** Initialize
         *
         * Valid data layouts:
         * - Any
         *
         * Valid data type configurations:
         * |lhs            |rhs                |dst            |
         * |:--------------|:------------------|:--------------|
         * |F32            |F32                |F32            |
         * |F16            |F16                |F16            |
         * |BFLOAT16       |BFLOAT16           |BFLOAT16       |
         * |QASYMM8_SIGNED |QASYMM8_SIGNED     |QASYMM8_SIGNED |
         * |QASYMM8        |QASYMM8            |QASYMM8        |
         *
         * @param[in]  lhs      Left-hand side tensor info. Data types supported: F16/F32/QASYMM8_SIGNED/QASYMM8.
         * @param[in]  rhs      Right-hand side tensor info. Data types supported: same as @p lhs.
         * @param[out] dst      Output tensor to store the result of the batched matrix multiplication. Data types supported: same as @p lhs / @p rhs.
         * @param[in]  info     Contains MatMul operation information described in @ref MatMulInfo.
         * @param[in]  settings Contains flags for function level settings i.e fast math
         * @param[in]  act_info (Optional) Contains activation function and lower and upper bound values for bounded activation functions.
         */
        void configure(BIITensor *lhs,
                       BIITensor *rhs,
                       BIITensor *dst,
                       const BIMatMulInfo &info,
                       const BICpuMatMulSettings &settings,
                       const BIActivationLayerInfo &act_info = BIActivationLayerInfo());

        void dynamic_configure(BIITensor *lhs,
                               BIITensor *rhs,
                               BIITensor *dst) const;

        static BIStatus validate(const BIITensorInfo *lhs,
                                 const BIITensorInfo *rhs,
                                 const BIITensorInfo *dst,
                                 const BIMatMulInfo &info,
                                 const BICpuMatMulSettings &settings,
                                 const BIActivationLayerInfo &act_info = BIActivationLayerInfo());

        void run();

    private:
        struct Impl;
        std::unique_ptr<Impl> _impl;
    };
}
