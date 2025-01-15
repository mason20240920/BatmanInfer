//
// Created by Mason on 2025/1/15.
//

#include <runtime/neon/functions/bi_ne_slice.hpp>

#include <data/core/bi_i_tensor.hpp>
#include <data/core/bi_types.hpp>
#include <data/core/utils/helpers/bi_tensor_transform.h>
#include <data/core/bi_vlidate.hpp>

#include <common/utils/bi_log.hpp>
#include <data/core/neon/kernels/bi_ne_stride_slice_kernel.hpp>

namespace BatmanInfer {
    namespace experimental {
        void BINESlice::configure(const BIITensorInfo *input,
                                  BIITensorInfo *output,
                                  const BICoordinates &starts,
                                  const BICoordinates &ends) {
            BI_COMPUTE_ERROR_ON_NULLPTR(input);
//            BI_COMPUTE_LOG_PARAMS(input, output, starts, ends);

            // Get absolute end coordinates
            const int32_t slice_end_mask = BatmanInfer::helpers::tensor_transform::construct_slice_end_mask(ends);

            auto k = std::make_unique<BINEStridedSliceKernel>();
            k->configure(input, output, starts, ends, BiStrides(), 0, slice_end_mask, 0);
            _kernel = std::move(k);
        }

        BIStatus BINESlice::validate(const BIITensorInfo *input,
                                     const BIITensorInfo *output,
                                     const BICoordinates &starts,
                                     const BICoordinates &ends) {
            BI_COMPUTE_RETURN_ERROR_ON_NULLPTR(input);

            // Check start dimensions for being non-negative
            BI_COMPUTE_RETURN_ERROR_ON(
                    std::any_of(starts.cbegin(), starts.cbegin() + starts.num_dimensions(),
                                [](int i) { return i < 0; }));

            // Get absolute end coordinates
            const int32_t slice_end_mask = BatmanInfer::helpers::tensor_transform::construct_slice_end_mask(ends);

            return BINEStridedSliceKernel::validate(input, output, starts, ends, BiStrides(), 0, slice_end_mask, 0);
        }
    } // namespace experimental

    struct BINESlice::Impl {
        const BIITensor *src{nullptr};
        BIITensor *dst{nullptr};
        std::unique_ptr<experimental::BINESlice> op{nullptr};
    };

    BINESlice::BINESlice() : _impl(std::make_unique<Impl>()) {
    }

    BINESlice::BINESlice(BINESlice &&) = default;

    BINESlice &BINESlice::operator=(BINESlice &&) = default;

    BINESlice::~BINESlice() = default;

    BIStatus BINESlice::validate(const BIITensorInfo *input,
                                 const BIITensorInfo *output,
                                 const BICoordinates &starts,
                                 const BICoordinates &ends) {
        return experimental::BINESlice::validate(input, output, starts, ends);
    }

    void BINESlice::configure(const BIITensor *input, BIITensor *output, const BICoordinates &starts,
                              const BICoordinates &ends) {
        _impl->src = input;
        _impl->dst = output;
        _impl->op = std::make_unique<experimental::BINESlice>();
        _impl->op->configure(input->info(), output->info(), starts, ends);
    }

    void BINESlice::run() {
        BIITensorPack pack;
        pack.add_tensor(BITensorType::ACL_SRC, _impl->src);
        pack.add_tensor(BITensorType::ACL_DST, _impl->dst);
        _impl->op->run(pack);
    }
}