//
// Created by Mason on 2025/1/16.
//

#include <runtime/neon/functions/bi_ne_elementwise_operations.hpp>

#include <data/core/bi_i_tensor.hpp>
#include <data/core/bi_vlidate.hpp>

#include <cpu/operators/bi_cpu_elementwise.hpp>

namespace BatmanInfer {
    struct BINEElementwiseMax::Impl {
        const BIITensor *src_0{nullptr};
        const BIITensor *src_1{nullptr};
        BIITensor *dst{nullptr};
        std::unique_ptr<cpu::BICpuElementwiseMax> op{nullptr};
    };

    BINEElementwiseMax::BINEElementwiseMax() : _impl(std::make_unique<Impl>()) {
    }

    BINEElementwiseMax::BINEElementwiseMax(BINEElementwiseMax &&) = default;

    BINEElementwiseMax &BINEElementwiseMax::operator=(BINEElementwiseMax &&) = default;

    BINEElementwiseMax::~BINEElementwiseMax() = default;

    void BINEElementwiseMax::configure(BIITensor *input1, BIITensor *input2, BIITensor *output,
                                       const BIActivationLayerInfo &act_info) {
        BI_COMPUTE_UNUSED(act_info);
        _impl->src_0 = input1;
        _impl->src_1 = input2;
        _impl->dst = output;
        _impl->op = std::make_unique<cpu::BICpuElementwiseMax>();
        _impl->op->configure(input1->info(), input2->info(), output->info());
    }

    BIStatus BINEElementwiseMax::validate(const BIITensorInfo *input1,
                                          const BIITensorInfo *input2,
                                          const BIITensorInfo *output,
                                          const BIActivationLayerInfo &act_info) {
        BI_COMPUTE_RETURN_ERROR_ON(act_info.enabled());
        return cpu::BICpuElementwiseMax::validate(input1, input2, output);
    }

    void BINEElementwiseMax::run() {
        BIITensorPack pack;
        pack.add_tensor(BITensorType::ACL_SRC_0, _impl->src_0);
        pack.add_tensor(BITensorType::ACL_SRC_1, _impl->src_1);
        pack.add_tensor(BITensorType::ACL_DST, _impl->dst);
        _impl->op->run(pack);
    }

    struct BINEElementwiseMin::Impl {
        const BIITensor *src_0{nullptr};
        const BIITensor *src_1{nullptr};
        BIITensor *dst{nullptr};
        std::unique_ptr<cpu::BICpuElementwiseMin> op{nullptr};
    };

    BINEElementwiseMin::BINEElementwiseMin() : _impl(std::make_unique<Impl>()) {
    }

    BINEElementwiseMin::BINEElementwiseMin(BINEElementwiseMin &&) = default;

    BINEElementwiseMin &BINEElementwiseMin::operator=(BINEElementwiseMin &&) = default;

    BINEElementwiseMin::~BINEElementwiseMin() = default;

    void BINEElementwiseMin::configure(BIITensor *input1, BIITensor *input2, BIITensor *output,
                                       const BIActivationLayerInfo &act_info) {
        BI_COMPUTE_UNUSED(act_info);
        _impl->src_0 = input1;
        _impl->src_1 = input2;
        _impl->dst = output;
        _impl->op = std::make_unique<cpu::BICpuElementwiseMin>();
        _impl->op->configure(input1->info(), input2->info(), output->info());
    }

    BIStatus BINEElementwiseMin::validate(const BIITensorInfo *input1,
                                          const BIITensorInfo *input2,
                                          const BIITensorInfo *output,
                                          const BIActivationLayerInfo &act_info) {
        BI_COMPUTE_RETURN_ERROR_ON(act_info.enabled());
        return cpu::BICpuElementwiseMin::validate(input1, input2, output);
    }

    void BINEElementwiseMin::run() {
        BIITensorPack pack;
        pack.add_tensor(BITensorType::ACL_SRC_0, _impl->src_0);
        pack.add_tensor(BITensorType::ACL_SRC_1, _impl->src_1);
        pack.add_tensor(BITensorType::ACL_DST, _impl->dst);
        _impl->op->run(pack);
    }

    struct BINEElementwiseSquaredDiff::Impl {
        const BIITensor *src_0{nullptr};
        const BIITensor *src_1{nullptr};
        BIITensor *dst{nullptr};
        std::unique_ptr<cpu::BICpuElementwiseSquaredDiff> op{nullptr};
    };

    BINEElementwiseSquaredDiff::BINEElementwiseSquaredDiff() : _impl(std::make_unique<Impl>()) {
    }

    BINEElementwiseSquaredDiff::BINEElementwiseSquaredDiff(BINEElementwiseSquaredDiff &&) = default;

    BINEElementwiseSquaredDiff &BINEElementwiseSquaredDiff::operator=(BINEElementwiseSquaredDiff &&) = default;

    BINEElementwiseSquaredDiff::~BINEElementwiseSquaredDiff() = default;

    void BINEElementwiseSquaredDiff::configure(BIITensor *input1,
                                               BIITensor *input2,
                                               BIITensor *output,
                                               const BIActivationLayerInfo &act_info) {
        BI_COMPUTE_UNUSED(act_info);
        _impl->src_0 = input1;
        _impl->src_1 = input2;
        _impl->dst = output;
        _impl->op = std::make_unique<cpu::BICpuElementwiseSquaredDiff>();
        _impl->op->configure(input1->info(), input2->info(), output->info());
    }

    BIStatus BINEElementwiseSquaredDiff::validate(const BIITensorInfo *input1,
                                                  const BIITensorInfo *input2,
                                                  const BIITensorInfo *output,
                                                  const BIActivationLayerInfo &act_info) {
        BI_COMPUTE_RETURN_ERROR_ON(act_info.enabled());
        return cpu::BICpuElementwiseSquaredDiff::validate(input1, input2, output);
    }

    void BINEElementwiseSquaredDiff::run() {
        BIITensorPack pack;
        pack.add_tensor(BITensorType::ACL_SRC_0, _impl->src_0);
        pack.add_tensor(BITensorType::ACL_SRC_1, _impl->src_1);
        pack.add_tensor(BITensorType::ACL_DST, _impl->dst);
        _impl->op->run(pack);
    }

    struct BINEElementwiseDivision::Impl {
        const BIITensor *src_0{nullptr};
        const BIITensor *src_1{nullptr};
        BIITensor *dst{nullptr};
        std::unique_ptr<cpu::BICpuElementwiseDivision> op{nullptr};
    };

    BINEElementwiseDivision::BINEElementwiseDivision() : _impl(std::make_unique<Impl>()) {
    }

    BINEElementwiseDivision::BINEElementwiseDivision(BINEElementwiseDivision &&) = default;

    BINEElementwiseDivision &BINEElementwiseDivision::operator=(BINEElementwiseDivision &&) = default;

    BINEElementwiseDivision::~BINEElementwiseDivision() = default;

    void BINEElementwiseDivision::configure(BIITensor *input1,
                                            BIITensor *input2,
                                            BIITensor *output,
                                            const BIActivationLayerInfo &act_info) {
        BI_COMPUTE_UNUSED(act_info);
        _impl->src_0 = input1;
        _impl->src_1 = input2;
        _impl->dst = output;
        _impl->op = std::make_unique<cpu::BICpuElementwiseDivision>();
        _impl->op->configure(input1->info(), input2->info(), output->info());
    }

    BIStatus BINEElementwiseDivision::validate(const BIITensorInfo *input1,
                                               const BIITensorInfo *input2,
                                               const BIITensorInfo *output,
                                               const BIActivationLayerInfo &act_info) {
        BI_COMPUTE_RETURN_ERROR_ON(act_info.enabled());
        return cpu::BICpuElementwiseDivision::validate(input1, input2, output);
    }

    void BINEElementwiseDivision::run() {
        BIITensorPack pack;
        pack.add_tensor(BITensorType::ACL_SRC_0, _impl->src_0);
        pack.add_tensor(BITensorType::ACL_SRC_1, _impl->src_1);
        pack.add_tensor(BITensorType::ACL_DST, _impl->dst);
        _impl->op->run(pack);
    }

    struct BINEElementwisePower::Impl {
        const BIITensor *src_0{nullptr};
        const BIITensor *src_1{nullptr};
        BIITensor *dst{nullptr};
        std::unique_ptr<cpu::BICpuElementwisePower> op{nullptr};
    };

    BINEElementwisePower::BINEElementwisePower() : _impl(std::make_unique<Impl>()) {
    }

    BINEElementwisePower::BINEElementwisePower(BINEElementwisePower &&) = default;

    BINEElementwisePower &BINEElementwisePower::operator=(BINEElementwisePower &&) = default;

    BINEElementwisePower::~BINEElementwisePower() = default;

    void BINEElementwisePower::configure(BIITensor *input1,
                                         BIITensor *input2,
                                         BIITensor *output,
                                         const BIActivationLayerInfo &act_info) {
        BI_COMPUTE_UNUSED(act_info);
        _impl->src_0 = input1;
        _impl->src_1 = input2;
        _impl->dst = output;
        _impl->op = std::make_unique<cpu::BICpuElementwisePower>();
        _impl->op->configure(input1->info(), input2->info(), output->info());
    }

    BIStatus BINEElementwisePower::validate(const BIITensorInfo *input1,
                                            const BIITensorInfo *input2,
                                            const BIITensorInfo *output,
                                            const BIActivationLayerInfo &act_info) {
        BI_COMPUTE_RETURN_ERROR_ON(act_info.enabled());
        return cpu::BICpuElementwisePower::validate(input1, input2, output);
    }

    void BINEElementwisePower::run() {
        BIITensorPack pack;
        pack.add_tensor(BITensorType::ACL_SRC_0, _impl->src_0);
        pack.add_tensor(BITensorType::ACL_SRC_1, _impl->src_1);
        pack.add_tensor(BITensorType::ACL_DST, _impl->dst);
        _impl->op->run(pack);
    }

    template<ComparisonOperation COP>
    struct BINEElementwiseComparisonStatic<COP>::Impl {
        const BIITensor *src_0{nullptr};
        const BIITensor *src_1{nullptr};
        BIITensor *dst{nullptr};
        std::unique_ptr<cpu::BICpuElementwiseComparisonStatic<COP>> op{nullptr};
    };

    template<ComparisonOperation COP>
    BINEElementwiseComparisonStatic<COP>::BINEElementwiseComparisonStatic() : _impl(std::make_unique<Impl>()) {
    }

    template<ComparisonOperation COP>
    BINEElementwiseComparisonStatic<COP>::BINEElementwiseComparisonStatic(BINEElementwiseComparisonStatic &&) = default;

    template<ComparisonOperation COP>
    BINEElementwiseComparisonStatic<COP> &
    BINEElementwiseComparisonStatic<COP>::operator=(BINEElementwiseComparisonStatic &&) = default;

    template<ComparisonOperation COP>
    BINEElementwiseComparisonStatic<COP>::~BINEElementwiseComparisonStatic() = default;

    template<ComparisonOperation COP>
    void BINEElementwiseComparisonStatic<COP>::configure(BIITensor *input1, BIITensor *input2, BIITensor *output) {
        _impl->src_0 = input1;
        _impl->src_1 = input2;
        _impl->dst = output;
        _impl->op = std::make_unique<cpu::BICpuElementwiseComparisonStatic<COP>>();
        _impl->op->configure(input1->info(), input2->info(), output->info());
    }

    template<ComparisonOperation COP>
    BIStatus BINEElementwiseComparisonStatic<COP>::validate(const BIITensorInfo *input1,
                                                            const BIITensorInfo *input2,
                                                            const BIITensorInfo *output) {
        return cpu::BICpuElementwiseComparisonStatic<COP>::validate(input1, input2, output);
    }

    template<ComparisonOperation COP>
    void BINEElementwiseComparisonStatic<COP>::run() {
        BIITensorPack pack;
        pack.add_tensor(BITensorType::ACL_SRC_0, _impl->src_0);
        pack.add_tensor(BITensorType::ACL_SRC_1, _impl->src_1);
        pack.add_tensor(BITensorType::ACL_DST, _impl->dst);
        _impl->op->run(pack);
    }

    struct BINEElementwiseComparison::Impl {
        const BIITensor *src_0{nullptr};
        const BIITensor *src_1{nullptr};
        BIITensor *dst{nullptr};
        std::unique_ptr<cpu::BICpuElementwiseComparison> op{nullptr};
    };

    BINEElementwiseComparison::BINEElementwiseComparison() : _impl(std::make_unique<Impl>()) {
    }

    BINEElementwiseComparison::BINEElementwiseComparison(BINEElementwiseComparison &&) = default;

    BINEElementwiseComparison &BINEElementwiseComparison::operator=(BINEElementwiseComparison &&) = default;

    BINEElementwiseComparison::~BINEElementwiseComparison() = default;

    void BINEElementwiseComparison::configure(BIITensor *input1, BIITensor *input2, BIITensor *output,
                                              ComparisonOperation op) {
        _impl->src_0 = input1;
        _impl->src_1 = input2;
        _impl->dst = output;
        _impl->op = std::make_unique<cpu::BICpuElementwiseComparison>();
        _impl->op->configure(input1->info(), input2->info(), output->info(), op);
    }

    BIStatus BINEElementwiseComparison::validate(const BIITensorInfo *input1,
                                                 const BIITensorInfo *input2,
                                                 const BIITensorInfo *output,
                                                 ComparisonOperation op) {
        return cpu::BICpuElementwiseComparison::validate(input1, input2, output, op);
    }

    void BINEElementwiseComparison::run() {
        BIITensorPack pack;
        pack.add_tensor(BITensorType::ACL_SRC_0, _impl->src_0);
        pack.add_tensor(BITensorType::ACL_SRC_1, _impl->src_1);
        pack.add_tensor(BITensorType::ACL_DST, _impl->dst);
        _impl->op->run(pack);
    }

// Supported Specializations
    template
    class BINEElementwiseComparisonStatic<ComparisonOperation::Equal>;

    template
    class BINEElementwiseComparisonStatic<ComparisonOperation::NotEqual>;

    template
    class BINEElementwiseComparisonStatic<ComparisonOperation::Greater>;

    template
    class BINEElementwiseComparisonStatic<ComparisonOperation::GreaterEqual>;

    template
    class BINEElementwiseComparisonStatic<ComparisonOperation::Less>;

    template
    class BINEElementwiseComparisonStatic<ComparisonOperation::LessEqual>;
}