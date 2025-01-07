//
// Created by Mason on 2025/1/7.
//

#ifndef BATMANINFER_BI_NE_TRANSPOSE_HPP
#define BATMANINFER_BI_NE_TRANSPOSE_HPP

#include <data/core/bi_types.hpp>
#include <runtime/bi_i_function.hpp>

#include <memory>

namespace BatmanInfer {
    // 前向声明
    class BIITensor;

    class BIITensorInfo;

    /**
     * 基本的方法运行 cpu::kernels::BICpuTransposeKernel
     */
    class BINETranspose : public BIIFunction {
    public:
        BINETranspose();

        ~BINETranspose();

        BINETranspose(const BINETranspose &) = delete;

        BINETranspose(BINETranspose &&) = default;

        BINETranspose &operator=(const BINETranspose &) = delete;

        BINETranspose &operator=(BINETranspose &&) = default;

        /**
         * Initialise the kernel's inputs and output
         *
         * Valid data layouts:
         * - All
         *
         * Valid data type configurations:
         * |src    |dst    |
         * |:------|:------|
         * |All    |All    |
         *
         * @param input
         * @param output
         */
        void configure(const BIITensor *input, BIITensor *output);

        /**
         * Static function to check if given info will lead to a valid configuration of @ref BINETranspose
         * @param input
         * @param output
         * @return
         */
        static BIStatus validate(const BIITensorInfo *input, const BIITensorInfo *output);

        void run() override;

    private:
        struct Impl;
        std::unique_ptr<Impl> _impl;
    };
}

#endif //BATMANINFER_BI_NE_TRANSPOSE_HPP
