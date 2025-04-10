//
// Created by Mason on 2025/4/10.
//

#include <cpu/kernels/BINEReductionOperationKernel.hpp>


#include <data/core/bi_coordinates.hpp>
#include <data/core/bi_helpers.hpp>
#include <data/core/bi_i_tensor.hpp>
#include <data/core/bi_tensor_info.hpp>
#include <data/core/bi_utils.hpp>
#include <data/core/utils/misc/bi_shape_calculator.hpp>
#include <data/core/bi_vlidate.hpp>

#include <common/bi_registers.hpp>
#include <data/core/cpp/bi_cpp_validate.hpp>
#include <data/core/helpers/bi_auto_configuration.hpp>
#include <data/core/helpers/bi_window_helpers.hpp>
#include <data/core/neon/bi_i_ne_kernel.hpp>
#include <data/core/neon/wrapper/wrapper.hpp>
#include <cpu/kernels/reduction_layer/generic/neon/list.hpp>

namespace BatmanInfer {
    void BINEReductionOperationKernel::reduce_op() {
        const bool is_complex = (_input->info()->num_channels() == 2);

        if (is_complex) {
            switch (_reduction_axis) {
                case 2:
                    switch (_input->info()->data_type()) {
                        case BIDataType::F32: {
                            switch (_op) {
                                case BIReductionOperation::SUM:
                                    _func = REGISTER_FP32_NEON(cpu::reduce_RedOpYZW_complex_reduceZ_float32_4_2_SUM);
                                    break;
                                default:
                                    BI_COMPUTE_ERROR("Not supported");
                                    break;
                            }
                            break;
                        }
                        default: {
                            BI_COMPUTE_ERROR("Not supported");
                            break;
                        }
                    }
                    break;
                default: {
                    BI_COMPUTE_ERROR("Not supported");
                    break;
                }
            }
            return;
        }

        switch (_reduction_axis) {
            case 0: {
                switch (_input->info()->data_type()) {
                    case BIDataType::QASYMM8: {
                        _func = REGISTER_QASYMM8_NEON(cpu::reduce_RedOpX_reduceX_qasymm8);
                        break;
                    }
                    case BIDataType::QASYMM8_SIGNED: {
                        _func = REGISTER_QASYMM8_SIGNED_NEON(cpu::reduce_RedOpX_reduceX_qasymm8_signed);
                        break;
                    }
#ifdef BI_COMPUTE_ENABLE_FP16
                    case BIDataType::F16: {
                        _func = REGISTER_FP16_NEON(cpu::reduce_RedOpX_reduceX_float16_8);
                        break;
                    }
#endif // BI_COMPUTE_ENABLE_FP16
                    case BIDataType::F32: {
                        _func = REGISTER_FP32_NEON(cpu::reduce_RedOpX_reduceX_float32_4);
                        break;
                    }
                    case BIDataType::S32: {
                        _func = REGISTER_INTEGER_NEON(cpu::reduce_RedOpX_reduceX_S32_4);
                        break;
                    }
                    default: {
                        BI_COMPUTE_ERROR("Not supported");
                        break;
                    }
                }
                break;
            }
            case 1: {
                switch (_input->info()->data_type()) {
                    case BIDataType::QASYMM8: {
                        _func = REGISTER_QASYMM8_NEON(cpu::reduce_RedOpYZW_reduceY_qasymm8);
                        break;
                    }
                    case BIDataType::QASYMM8_SIGNED: {
                        _func = REGISTER_QASYMM8_SIGNED_NEON(cpu::reduce_RedOpYZW_reduceY_qasymm8_signed);
                        break;
                    }
#ifdef BI_COMPUTE_ENABLE_FP16
                    case BIDataType::F16: {
                        _func = REGISTER_FP16_NEON(cpu::reduce_RedOpYZW_reduceY_float16_8);
                        break;
                    }
#endif // BI_COMPUTE_ENABLE_FP16
                    case BIDataType::F32: {
                        _func = REGISTER_FP32_NEON(cpu::reduce_RedOpYZW_reduceY_float32_4);
                        break;
                    }
                    case BIDataType::S32: {
                        _func = REGISTER_INTEGER_NEON(cpu::reduce_RedOpYZW_reduceY_S32_4);
                        break;
                    }
                    default: {
                        BI_COMPUTE_ERROR("Not supported");
                        break;
                    }
                }
                break;
            }
            case 2: {
                switch (_input->info()->data_type()) {
                    case BIDataType::QASYMM8: {
                        _func = REGISTER_QASYMM8_NEON(cpu::reduce_RedOpYZW_reduceZ_qasymm8);
                        break;
                    }
                    case BIDataType::QASYMM8_SIGNED: {
                        _func = REGISTER_QASYMM8_SIGNED_NEON(cpu::reduce_RedOpYZW_reduceZ_qasymm8_signed);
                        break;
                    }
#ifdef BI_COMPUTE_ENABLE_FP16
                    case BIDataType::F16: {
                        _func = REGISTER_FP16_NEON(cpu::reduce_RedOpYZW_reduceZ_float16_8);
                        break;
                    }
#endif // BI_COMPUTE_ENABLE_FP16
                    case BIDataType::F32: {
                        _func = REGISTER_FP32_NEON(cpu::reduce_RedOpYZW_reduceZ_float32_4);
                        break;
                    }
                    case BIDataType::S32: {
                        _func = REGISTER_INTEGER_NEON(cpu::reduce_RedOpYZW_reduceZ_S32_4);
                        break;
                    }
                    default: {
                        std::cout << int(_input->info()->data_type()) << std::endl;
                        BI_COMPUTE_ERROR("Not supported");
                        break;
                    }
                }
                break;
            }
            case 3: {
                switch (_input->info()->data_type()) {
                    case BIDataType::QASYMM8: {
                        _func = REGISTER_QASYMM8_NEON(cpu::reduce_RedOpYZW_reduceW_qasymm8);
                        break;
                    }
                    case BIDataType::QASYMM8_SIGNED: {
                        _func = REGISTER_QASYMM8_SIGNED_NEON(cpu::reduce_RedOpYZW_reduceW_qasymm8_signed);
                        break;
                    }
#ifdef BI_COMPUTE_ENABLE_FP16
                    case BIDataType::F16: {
                        _func = REGISTER_FP16_NEON(cpu::reduce_RedOpYZW_reduceW_float16_8);
                        break;
                    }
#endif // BI_COMPUTE_ENABLE_FP16
                    case BIDataType::F32: {
                        _func = REGISTER_FP32_NEON(cpu::reduce_RedOpYZW_reduceW_float32_4);
                        break;
                    }
                    case BIDataType::S32: {
                        _func = REGISTER_INTEGER_NEON(cpu::reduce_RedOpYZW_reduceW_S32_4);
                        break;
                    }
                    default: {
                        BI_COMPUTE_ERROR("Not supported");
                        break;
                    }
                }
                break;
            }
            default: {
                BI_COMPUTE_ERROR("Unsupported reduction axis");
                break;
            }
        }
    }

    BIStatus validate_arguments(const BIITensorInfo *input, const BIITensorInfo *output, unsigned int axis,
                                BIReductionOperation op) {
        BI_COMPUTE_UNUSED(op);

        BI_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
        BI_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(input);

        if (input->num_channels() == 1) {
            BI_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, BIDataType::QASYMM8_SIGNED,
                                                                BIDataType::QASYMM8,
                                                                BIDataType::S32, BIDataType::F16, BIDataType::F32);
        } else {
            BI_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 2, BIDataType::F32);
            BI_COMPUTE_RETURN_ERROR_ON(op != BIReductionOperation::SUM);
            BI_COMPUTE_RETURN_ERROR_ON(axis != 2);
        }

        BI_COMPUTE_RETURN_ERROR_ON_MSG(axis >= BITensorShape::num_max_dimensions,
                                       "Reduction axis greater than max number of dimensions");
        BI_COMPUTE_RETURN_ERROR_ON_MSG(axis > 3, "Unsupported reduction axis");

        if (output->total_size() != 0) {
            bool is_arg_min_max = (op == BIReductionOperation::ARG_IDX_MAX || op == BIReductionOperation::ARG_IDX_MIN);
            if (!is_arg_min_max) {
                BI_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
                BI_COMPUTE_RETURN_ERROR_ON(input->num_channels() != output->num_channels());
            } else {
                BI_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, BIDataType::U32, BIDataType::S32);
            }

            const BITensorShape output_shape =
                    BatmanInfer::misc::shape_calculator::compute_reduced_shape(input->tensor_shape(), axis);
            const BITensorInfo tensor_info_reshaped = input->clone()->set_tensor_shape(output_shape);
            BI_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(output, &tensor_info_reshaped);
        }

        return BIStatus{};
    }

    BINEReductionOperationKernel::BINEReductionOperationKernel()
        : _func(nullptr), _input(nullptr), _output(nullptr), _reduction_axis(0), _op(BIReductionOperation::SUM_SQUARE) {
    }

    void BINEReductionOperationKernel::configure(const BIITensor *input,
                                                 BIITensor *output,
                                                 unsigned int axis,
                                                 BIReductionOperation op) {
        BI_COMPUTE_ERROR_ON_NULLPTR(input, output);

        BI_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info(), axis, op));

        _input = input;
        _output = output;
        _op = op;
        _reduction_axis = axis;

        // Configure kernel window
        BIWindow win = calculate_max_window(*input->info(), BISteps());
        BIINEKernel::configure(win);

        // Calculate output shape and set if empty
        const BITensorShape output_shape =
                BatmanInfer::misc::shape_calculator::compute_reduced_shape(input->info()->tensor_shape(), axis);
        // Output auto initialization if not yet initialized
        const bool is_arg_min_max = (op == BIReductionOperation::ARG_IDX_MIN || op ==
                                     BIReductionOperation::ARG_IDX_MAX);
        BIDataType output_data_type = is_arg_min_max ? BIDataType::S32 : input->info()->data_type();
        auto_init_if_empty(*output->info(), input->info()
                           ->clone()
                           ->set_tensor_shape(output_shape)
                           .set_data_type(output_data_type)
                           .reset_padding()
                           .set_is_resizable(true));
        // Determine the reduction function
        BINEReductionOperationKernel::reduce_op();
    }

    BIStatus BINEReductionOperationKernel::validate(const BIITensorInfo *input,
                                                    const BIITensorInfo *output,
                                                    unsigned int axis,
                                                    BIReductionOperation op) {
        BI_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, axis, op));

        return BIStatus{};
    }

    void BINEReductionOperationKernel::run(const BIWindow &window, const ThreadInfo &info) {
        BI_COMPUTE_UNUSED(info);
        BI_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
        BI_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(BIINEKernel::window(), window);

        (*_func)(window, _input, _output, _op);
    }
}
