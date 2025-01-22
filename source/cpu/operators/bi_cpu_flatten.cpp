//
// Created by Mason on 2025/1/22.
//

#include <cpu/operators/bi_cpu_flatten.hpp>
#include <cpu/operators/bi_cpu_reshape.hpp>
#include <data/core/bi_i_tensor_info.hpp>

#include <common/utils/bi_log.hpp>


namespace BatmanInfer {
    namespace cpu {
        BICpuFlatten::BICpuFlatten() : _reshape(nullptr) {

        }

        BICpuFlatten::~BICpuFlatten() = default;

        void BICpuFlatten::configure(const BatmanInfer::BIITensorInfo *src,
                                     BatmanInfer::BIITensorInfo *dst) {
            BI_COMPUTE_LOG_PARAMS(src, dst);
            _reshape = std::make_unique<BICpuReshape>();
            _reshape->configure(src, dst);
        }

        BIStatus
        BICpuFlatten::validate(const BatmanInfer::BIITensorInfo *src, const BatmanInfer::BIITensorInfo *dst) {
            return BICpuReshape::validate(src, dst);
        }

        void BICpuFlatten::run(BatmanInfer::BIITensorPack &tensors) {
            _reshape->run(tensors);
        }
    }
}