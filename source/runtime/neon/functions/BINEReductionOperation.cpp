//
// Created by Mason on 2025/4/10.
//

#include <runtime/neon/functions/BINEReductionOperation.hpp>

#include <data/core/bi_helpers.hpp>
#include <data/core/utils/misc/bi_shape_calculator.hpp>
#include <runtime/neon/bi_ne_scheduler.hpp>

#include <common/utils/bi_log.hpp>
#include <data/core/helpers/bi_auto_configuration.hpp>
#include <cpu/kernels/BINEReductionOperationKernel.hpp>

namespace BatmanInfer {
    namespace {
        /** Define dimension to split the window
         *
         * @param[in] axis Reduction axis
         *
         * @return The dimension to split the window
         */
        size_t reduction_window_split_dimension(unsigned int axis) {
            switch (axis) {
                case 0:
                    return BIWindow::DimY;
                case 1:
                case 2:
                case 3:
                    return BIWindow::DimX;
                default:
                    BI_COMPUTE_ERROR("Unsupported reduction axis");
            }
        }
    } // namespace

    BINEReductionOperation::~BINEReductionOperation() = default;

    BINEReductionOperation::BINEReductionOperation(std::shared_ptr<BIIMemoryManager> memory_manager)
        : _memory_group(memory_manager),
          _reduction_kernel(),
          _reshape(),
          _output_internal(),
          _window_split(0),
          _reduction_axis(),
          _is_reshape_required(false) {
    }

    BIStatus BINEReductionOperation::validate(
        const BIITensorInfo *input, const BIITensorInfo *output, unsigned int axis, BIReductionOperation op,
        bool keep_dims) {
        BI_COMPUTE_RETURN_ERROR_ON_MSG(axis >= BITensorShape::num_max_dimensions,
                                       "Reduction axis greater than max number of dimensions");
        BI_COMPUTE_RETURN_ERROR_ON_MSG(axis > 3, "Unsupported reduction axis");
        BI_COMPUTE_RETURN_ERROR_ON_DYNAMIC_SHAPE(input, output);

        const auto is_reshape_required = !keep_dims;

        auto *output_internal = output;

        BITensorInfo info_before_reshape;

        if (is_reshape_required) {
            const BITensorInfo expected_output_shape = output->clone()->set_tensor_shape(
                misc::shape_calculator::compute_reduced_shape(input->tensor_shape(), axis, keep_dims));
            BI_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(&expected_output_shape, output);

            auto shape_before_reshape = input->tensor_shape();
            shape_before_reshape.set(axis, 1);

            const auto input_num_channles = input->num_channels();
            const auto input_qinfo = input->quantization_info();
            const auto is_arg_min_max = (op == BIReductionOperation::ARG_IDX_MAX) || (
                                            op == BIReductionOperation::ARG_IDX_MIN);
            const auto output_data_type = is_arg_min_max ? BIDataType::S32 : output->data_type();

            info_before_reshape.set_data_type(output_data_type)
                    .set_tensor_shape(shape_before_reshape)
                    .set_num_channels(input_num_channles)
                    .set_quantization_info(input_qinfo);

            output_internal = &info_before_reshape;
        }

        BI_COMPUTE_RETURN_ON_ERROR(BINEReductionOperationKernel::validate(input, output_internal, axis, op));

        if (is_reshape_required) {
            BI_COMPUTE_RETURN_ON_ERROR(BINEReshapeLayer::validate(output_internal, output));
        }

        return BIStatus{};
    }

    void BINEReductionOperation::configure(
        BIITensor *input, BIITensor *output, unsigned int axis, BIReductionOperation op, bool keep_dims) {
        BI_COMPUTE_ERROR_ON_NULLPTR(input, output);
        BI_COMPUTE_LOG_PARAMS(input, output, axis, op, keep_dims);

        _is_reshape_required = !keep_dims;

        auto *output_internal = output;
        const auto is_arg_min_max = (op == BIReductionOperation::ARG_IDX_MAX) || (
                                        op == BIReductionOperation::ARG_IDX_MIN);

        if (_is_reshape_required) {
            const auto output_internal_shape =
                    misc::shape_calculator::compute_reduced_shape(input->info()->tensor_shape(), axis);
            const auto output_external_shape =
                    misc::shape_calculator::compute_reduced_shape(
                        input->info()->tensor_shape(), axis, false);
            const auto output_data_type = is_arg_min_max ? BIDataType::S32 : input->info()->data_type();
            const auto num_channels = input->info()->num_channels();
            const auto qinfo = input->info()->quantization_info();

            _output_internal.allocator()->init(input->info()
                ->clone()
                ->set_data_type(output_data_type)
                .set_tensor_shape(output_internal_shape)
                .reset_padding()
                .set_is_resizable(true)
                .set_num_channels(num_channels)
                .set_quantization_info(qinfo));
            _memory_group.manage(&_output_internal);
            output_internal = &_output_internal;
            auto_init_if_empty(*output->info(), input->info()
                               ->clone()
                               ->set_data_type(output_data_type)
                               .set_tensor_shape(output_external_shape)
                               .reset_padding()
                               .set_is_resizable(true));
        }

        BI_COMPUTE_ERROR_THROW_ON(BINEReductionOperation::validate(input->info(), output->info(), axis, op, keep_dims));

        // Configure reduction kernel
        _reduction_kernel = std::make_unique<BINEReductionOperationKernel>();
        _reduction_kernel->configure(input, output_internal, axis, op);
        _window_split = reduction_window_split_dimension(axis);
        _reduction_axis = axis;

        if (_is_reshape_required) {
            _reshape.configure(output_internal, output);
            _output_internal.allocator()->allocate();
        }
    }

    void BINEReductionOperation::run() {
        BIMemoryGroupResourceScope scope_mg(_memory_group);
        BINEScheduler::get().schedule(_reduction_kernel.get(), _window_split);
        if (_is_reshape_required) {
            _reshape.run();
        }
    }
} // namespace BatmanInfer
