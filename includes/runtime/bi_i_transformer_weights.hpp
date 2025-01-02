//
// Created by Mason on 2025/1/2.
//

#ifndef BATMANINFER_BI_I_TRANSFORMER_WEIGHTS_HPP
#define BATMANINFER_BI_I_TRANSFORMER_WEIGHTS_HPP

#include <atomic>
#include <utility>

namespace BatmanInfer {
    // 前向声明
    class BIITensor;

    /**
     * @brief 权重张量变换接口
     *        为了区分不同的重排（reshape）函数，每个重排函数必须生成一个唯一的 ID
     *        我们通过以下方式使用一个无符号的 32 位值进行转换:
     *
     * 最低两位存储目标类型：
     * 00 -> Neon
     * 01 -> CL
     * 11 -> 未使用
     *
     * 接下来的五位存储重排函数的 ID：
     * 00000 -> FullyConnectedLayerReshapeWeights（全连接层重排权重）
     * 00001 -> ConvertFullyConnectedWeights（转换全连接权重）
     * 00010 -> ConvolutionLayerReshapeWeights（卷积层重排权重）
     * 00011 -> DepthwiseConvolutionLayerReshapeWeights（深度卷积层重排权重）
     * 00100 -> GEMMReshapeLHSMatrixKernel（GEMM 左矩阵重排内核）
     * 00101 -> GEMMReshapeRHSMatrixKernel（GEMM 右矩阵重排内核）
     *
     * 剩余的位用于识别特殊情况，例如汇编函数和重排内核中的额外参数。
     */
    class BIITransformWeights
    {
    public:
        BIITransformWeights() = default;

        virtual ~BIITransformWeights() = default;

        BIITransformWeights(const BIITransformWeights &) = delete;

        BIITransformWeights &operator=(const BIITransformWeights &) = delete;

        BIITransformWeights(BIITransformWeights &&other) {
            *this = std::move(other);
        }

        BIITransformWeights &operator=(BIITransformWeights &&other) {
            if (this != &other) {
                _num_refcount = other._num_refcount.load();
                _reshape_run = other._reshape_run;
            }
            return *this;
        }

        /**
         * @brief 获取转换权重的指针
         * @return  返回转换张量权重
         */
        virtual BIITensor *get_weights() = 0;

        /**
         * @brief 返回重排函数的独立的Uid
         * @return
         */
        virtual uint32_t uid() = 0;

        /**
         * @brief 运行重排函数
         */
        virtual void run() = 0;

        /**
         * @brief 释放转换权重内存
         */
        virtual void release() = 0;

        /**
         * @brief 增加对象的引用计数
         */
        void increase_refcount() {
            ++_num_refcount;
        }

        int32_t decrease_refcount() {
            return --_num_refcount;
        }

        bool is_reshape_run() const {
            return _reshape_run;
        }

    private:
        std::atomic<int32_t> _num_refcount{0};
        bool _reshape_run{false};
    };
}

#endif //BATMANINFER_BI_I_TRANSFORMER_WEIGHTS_HPP
