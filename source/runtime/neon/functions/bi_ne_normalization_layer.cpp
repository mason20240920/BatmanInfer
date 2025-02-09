//
// Created by Mason on 2025/2/9.
//

#include <runtime/neon/functions/bi_ne_normalization_layer.hpp>

#include <data/core/bi_error.h>
#include <data/core/bi_tensor_info.hpp>
#include <data/core/bi_types.hpp>
#include <data/core/bi_vlidate.hpp>
#include <runtime/neon/bi_ne_scheduler.hpp>

#include <common/utils/bi_log.hpp>
#include <cpu/kernels/bi_ne_normalization_layer_kernel.hpp>

namespace BatmanInfer {
    BINENormalizationLayer::~BINENormalizationLayer() = default;

    BINENormalizationLayer::BINENormalizationLayer(std::shared_ptr<BIIMemoryManager> memory_manager)
            : _memory_group(std::move(memory_manager)), _norm_kernel(), _multiply_f(), _input_squared() {
    }

    void
    BINENormalizationLayer::configure(const BIITensor *input, BIITensor *output,
                                      const BINormalizationLayerInfo &norm_info) {
        BI_COMPUTE_ERROR_ON_NULLPTR(input, output);
        BI_COMPUTE_LOG_PARAMS(input, output, norm_info);

        BITensorInfo tensor_info(input->info()->tensor_shape(), 1, input->info()->data_type());
        _input_squared.allocator()->init(tensor_info);

        // Manage intermediate buffers
        _memory_group.manage(&_input_squared);

        // Configure kernels
        _norm_kernel = std::make_unique<BINENormalizationLayerKernel>();
        _norm_kernel->configure(input, &_input_squared, output, norm_info);
        _multiply_f.configure(input, input, &_input_squared, 1.0f, BIConvertPolicy::SATURATE,
                              BIRoundingPolicy::TO_ZERO);

        // Allocate the tensor once the configure methods have been called
        _input_squared.allocator()->allocate();
    }

    BIStatus BINENormalizationLayer::validate(const BIITensorInfo *input,
                                              const BIITensorInfo *output,
                                              const BINormalizationLayerInfo &norm_info) {
        // Perform validation step
        BI_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);

        BI_COMPUTE_RETURN_ON_ERROR(BINENormalizationLayerKernel::validate(input, input, output, norm_info));
        BI_COMPUTE_RETURN_ON_ERROR(
                BINEPixelWiseMultiplication::validate(input, input, output, 1.0f, BIConvertPolicy::SATURATE,
                                                      BIRoundingPolicy::TO_ZERO));

        return BIStatus{};
    }

    void BINENormalizationLayer::run() {
        BIMemoryGroupResourceScope scope_mg(_memory_group);
        _multiply_f.run();
        BINEScheduler::get().schedule(_norm_kernel.get(), BIWindow::DimY);
    }
}