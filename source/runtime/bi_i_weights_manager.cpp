//
// Created by Mason on 2025/1/2.
//

#include <runtime/bi_i_weights_manager.hpp>

namespace BatmanInfer {

    BIIWeightsManager::BIIWeightsManager() : _managed_counter(), _managed_weights(), _managed_weights_parents() {

    }

    void BIIWeightsManager::manage(const BatmanInfer::BIITensor *weights, BatmanInfer::BIITransformWeights *parent) {
        if (!are_weights_managed(weights)) {
            _managed_weights[weights];
            _managed_counter[weights];
        } else {
            _managed_counter[weights].counter++;
        }

        // 如果权重是先前重塑函数的输出
        // 保存父节点的链接
        if (parent != nullptr) {
            if (_managed_weights_parents.find(weights) == _managed_weights_parents.end())
                _managed_weights_parents[weights] = parent;
        }
    }

    bool BIIWeightsManager::are_weights_managed(const BatmanInfer::BIITensor *weights) {
        return (_managed_weights.find(weights) != _managed_weights.end());
    }

    BIITensor *
    BIIWeightsManager::run(const BatmanInfer::BIITensor *weights, BatmanInfer::BIITransformWeights *weights_transform) {
        BI_COMPUTE_ERROR_ON_MSG(!are_weights_managed(weights), "Cannot run function. Weights are not managed");

        // 检查我是否与权重变换后的权重相同。如果相同，则不要运行重塑操作。
        auto item = _managed_weights.find(weights);
        bool perform_run{true};
        BIITensor *weights_tensor{nullptr};

        // 检查我是否已经拥有所请求的变换，并且是否已经运行了重塑函数。
        for (auto it: item->second) {
            if (it->is_reshape_run() && (it->uid() == weights_transform->uid())) {
                weights_tensor = it->get_weights();
                perform_run = false;
                break;
            }
        }

        if (perform_run) {
            weights_transform->run();
            weights_tensor = weights_transform->get_weights();
        }

        // 检查是否我们能够释放父节点的内存
        auto parent_item = _managed_weights_parents.find(weights);
        if (parent_item != _managed_weights_parents.end()) {
            int32_t refcount = parent_item->second->decrease_refcount();
            if (refcount == 0)
                parent_item->second->release();
        }

        // 检查高层级权重，如果所有转换信息被做完
        // 设置权重为无用
        if (_managed_weights_parents.find(weights) == _managed_weights_parents.end()) {
            bool mark_as_unused = true;
            for (auto it : item->second) {
                if (!it->is_reshape_run()) {
                    mark_as_unused = false;
                    break;
                }
            }

            if (mark_as_unused) {
                weights->mark_as_unused();
            }
        }

        return weights_tensor;
    }

    BIITensor *BIIWeightsManager::acquire(const BatmanInfer::BIITensor *weights,
                                          BatmanInfer::BIITransformWeights *weights_transform) {
        BI_COMPUTE_ERROR_ON_MSG(!are_weights_managed(weights), "Cannot acquire weights. Weights are not managed");

        BIITensor *transformer_weights{nullptr};
        auto item = _managed_weights.find(weights);

        // 检查我是否已经拥有请求的变换。如果有，
        // 增加变换权重对象的引用计数并
        // 重新使用张量
        for (auto it : item->second) {
            if (it->uid() == weights_transform->uid()) {
                transformer_weights = it->get_weights();
                it->increase_refcount();
                break;
            }
        }

        if (transformer_weights == nullptr) {
            transformer_weights = weights_transform->get_weights();
            weights_transform->increase_refcount();
            item->second.emplace_back(weights_transform);
        }

        // 管理权重并且保存链接到父结点
        manage(transformer_weights, weights_transform);

        return transformer_weights;
    }

    void BIIWeightsManager::release(const BatmanInfer::BIITensor *weights) {
        if (weights == nullptr || !are_weights_managed(weights))
            return;

        _managed_counter[weights].counter--;
        if (_managed_counter[weights].counter == 0 && _managed_counter[weights].is_unused)
            weights->mark_as_unused();
    }

    void BIIWeightsManager::pre_mark_as_unused(const BatmanInfer::BIITensor *weights) {
        if (weights == nullptr || !are_weights_managed(weights))
            return;

        _managed_counter[weights].is_unused = true;
    }

}