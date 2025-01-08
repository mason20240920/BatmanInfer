//
// Created by Mason on 2025/1/8.
//

#ifndef BATMANINFER_BI_NE_RESHAPE_LAYER_HPP
#define BATMANINFER_BI_NE_RESHAPE_LAYER_HPP

#include <data/core/bi_types.hpp>
#include <runtime/bi_i_function.hpp>
#include <runtime/neon/bi_i_ne_operator.hpp>
#include <runtime/bi_types.hpp>
#include <data/core/bi_tensor_info.hpp>

namespace BatmanInfer {
    /**
     * 基本的类: 运行cpu::kernels::BICpuReshapeKernel
     */
    class BINEReshapeLayer : public BIIFunction {
    public:
        BINEReshapeLayer();

        ~BINEReshapeLayer();

        BINEReshapeLayer(const BINEReshapeLayer &) = delete;

        BINEReshapeLayer(BINEReshapeLayer &&);

        BINEReshapeLayer &operator=(const BINEReshapeLayer &) = delete;

        BINEReshapeLayer &operator=(BINEReshapeLayer &&);

        /**
         * 初始化内核的输入和输出
         *
         * Valid data layouts:
         * - All
         *
         * Valid data type configurations:
         * |src    |dst    |
         * |:------|:------|
         * |All    |All    |
         * @param input 输入张量，数据类型支持: All
         * @param output 输出张量，数据类型支持: All
         */
        void configure(const BIITensor *input, BIITensor *output);

        /**
         *
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

#endif //BATMANINFER_BI_NE_RESHAPE_LAYER_HPP
