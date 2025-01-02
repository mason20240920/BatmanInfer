//
// Created by Mason on 2025/1/2.
//

#ifndef BATMANINFER_BI_I_WEIGHTS_MANAGER_HPP
#define BATMANINFER_BI_I_WEIGHTS_MANAGER_HPP

#include <runtime/bi_i_transformer_weights.hpp>
#include <data/core/bi_i_tensor.hpp>

#include <map>

namespace BatmanInfer {
    /**
     * @brief 权重管理器接口用于处理权重转换。
     */
    class BIIWeightsManager {
    public:
        BIIWeightsManager();

        virtual ~BIIWeightsManager() = default;

        BIIWeightsManager(const BIIWeightsManager &) = delete;

        BIIWeightsManager &operator=(const BIIWeightsManager &) = delete;

        BIIWeightsManager (BIIWeightsManager &&) = default;

        BIIWeightsManager &operator=(BIIWeightsManager &&) = default;

        /**
         * @brief 开始管理权重张量
         * @param weights 指针指向权重张量
         * @param parent  父结点: 权重来自于之前的重排函数
         */
        void manage(const BIITensor *weights, BIITransformWeights *parent = nullptr);

        /**
         * @brief 运行重排函数
         * @param weights  指针指向我们想要重排的权重张量
         * @param weights_transform  权重转换对象
         * @return
         */
        BIITensor *run(const BIITensor *weights, BIITransformWeights *weights_transform);

        /**
         * @brief 获取所选权重的请求重塑张量
         * @param weights
         * @param weights_transform
         * @return
         */
        BIITensor *acquire(const BIITensor *weights, BIITransformWeights *weights_transform);

        /**
         * @brief Check if the weights are managed
         * @param weights
         * @return
         */
        bool are_weights_managed(const BIITensor *weights);

        /**
         * @brief 释放权重引用计数，并在达到0时标记为未使用。
         * @param weights
         */
        void release(const BIITensor *weights);

        /**
         * @brief 预先将权重标记为未使用。只有当计数器归零时，权重张量才会被标记为未使用。
         * @param weights
         */
        void pre_mark_as_unused(const BIITensor *weights);

    private:
        struct BICounterElement {
            bool  is_unused{false};
            std::atomic<int> counter{1};
        };

    private:
        std::map<const BIITensor *, std::vector<BIITransformWeights *>> _managed_weights;
        std::map<const BIITensor *, BICounterElement> _managed_counter;
        std::map<const BIITensor *, BIITransformWeights *> _managed_weights_parents;
    };
}

#endif //BATMANINFER_BI_I_WEIGHTS_MANAGER_HPP
