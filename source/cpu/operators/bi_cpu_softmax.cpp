//
// Created by Mason on 2025/1/18.
//

#include <cpu/operators/bi_cpu_softmax.hpp>

#include <data/core/bi_helpers.hpp>
#include <data/core/bi_tensor_info.hpp>
#include <data/core/utils/misc/bi_shape_calculator.hpp>
#include <data/core/bi_vlidate.hpp>
#include <runtime/neon/bi_ne_scheduler.hpp>

#include <common/utils/bi_log.hpp>
#include <data/core/helpers/bi_memory_helpers.hpp>
#include <data/core/helpers/bi_softmax_helpers.hpp>
#include <cpu/kernels/bi_cpu_softmax_kernel.hpp>
#include <cpu/utils/bi_cpu_aux_tensor_handler.hpp>

using namespace BatmanInfer::experimental;

namespace BatmanInfer {
    namespace cpu {
        BICpuSoftmaxGeneric::BICpuSoftmaxGeneric() : _softmax_kernel(), _tmp(), _aux_mem(InternalTensorIdx::COUNT) {
        }

        void
        BICpuSoftmaxGeneric::configure(const BIITensorInfo *src, BIITensorInfo *dst, float beta, int32_t axis,
                                       bool is_log) {
            // Perform validation step
            BI_COMPUTE_ERROR_ON_NULLPTR(src, dst);
            BI_COMPUTE_ERROR_THROW_ON(BICpuSoftmaxGeneric::validate(src, dst, beta, axis, is_log));
            BI_COMPUTE_LOG_PARAMS(src, dst, beta, axis);

            const unsigned int actual_axis =
                    static_cast<unsigned int>(wrap_around(axis, static_cast<int32_t>(src->num_dimensions())));

            _axis = actual_axis;

            const BIITensorInfo *tmp_input = src;

            BITensorInfo tensor_info_tmp;
            if (is_data_type_quantized_asymmetric(src->data_type())) {
                // Create intermediate tensors shapes
                const BITensorInfo input_info = tmp_input->clone()->reset_padding().set_is_resizable(true);
                tensor_info_tmp = input_info.clone()->set_data_type(BIDataType::F32);
            }

            // Init intermediate tensors
            _tmp = BITensorInfo(tensor_info_tmp);

            // Configure kernels
            auto sm = std::make_unique<kernels::BICpuSoftmaxKernel>();

            // Softmax 2D case
            sm->configure(tmp_input, dst, beta, is_log, actual_axis, &_tmp);

            _softmax_kernel = std::move(sm);

            if (_tmp.total_size() > 0) {
                _aux_mem[InternalTensorIdx::TMP] =
                        BIMemoryInfo(offset_int_vec(InternalTensorIdx::TMP), MemoryLifetime::Temporary,
                                     _tmp.total_size());
            }
        }

        BIStatus
        BICpuSoftmaxGeneric::validate(const BIITensorInfo *src, const BIITensorInfo *dst, float beta, int32_t axis,
                                      bool is_log) {
            // Perform validation step
            BI_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, dst);
            BI_COMPUTE_RETURN_ERROR_ON_MSG(src->num_dimensions() > 4, "Only up to 4 dimensions are supported");
            BI_COMPUTE_UNUSED(beta);
            BI_COMPUTE_RETURN_ERROR_ON(axis < static_cast<int32_t>(-src->num_dimensions()) ||
                                       static_cast<int32_t>(src->num_dimensions()) <= axis);

            // Create intermediate tensor info
            BITensorInfo tensor_info_tmp;

            if (is_data_type_quantized_asymmetric(src->data_type())) {
                tensor_info_tmp = src->clone()->set_data_type(BIDataType::F32).set_is_resizable(true);
            }
            const unsigned int actual_axis =
                    static_cast<unsigned int>(wrap_around(axis, static_cast<int32_t>(src->num_dimensions())));

            BI_COMPUTE_RETURN_ON_ERROR(
                    kernels::BICpuSoftmaxKernel::validate(src, dst, beta, actual_axis, is_log, &tensor_info_tmp));

            return BIStatus{};
        }

        void BICpuSoftmaxGeneric::run(BIITensorPack &tensors) {
            BI_COMPUTE_ERROR_ON_MSG(tensors.empty(), "No inputs provided");

            auto src = tensors.get_const_tensor(BITensorType::ACL_SRC);
            auto dst = tensors.get_tensor(BITensorType::ACL_DST);

            CpuAuxTensorHandler tmp(offset_int_vec(InternalTensorIdx::TMP), _tmp, tensors, true);

            BIITensorPack softmax_pack;

            softmax_pack = {{BITensorType::ACL_SRC_0, src},
                            {BITensorType::ACL_DST_0, dst},
                            {BITensorType::ACL_DST_1, tmp.get()}};

            if (_axis == 0) {
                BINEScheduler::get().schedule_op(_softmax_kernel.get(), BIWindow::DimY, _softmax_kernel->window(),
                                                 softmax_pack);
            } else {
                BINEScheduler::get().schedule_op(_softmax_kernel.get(), BIWindow::DimX, _softmax_kernel->window(),
                                                 softmax_pack);
            }
        }

        experimental::BIMemoryRequirements BICpuSoftmaxGeneric::workspace() const {
            return _aux_mem;
        }

    } // namespace cpu
}