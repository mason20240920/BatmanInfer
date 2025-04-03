//
// Created by Mason on 2025/4/3.
//

#include <cpu/kernels/BINESelectKernel.hpp>

#include <data/core/bi_error.h>
#include <data/core/bi_helpers.hpp>
#include <data/core/bi_i_tensor.hpp>
#include <data/core/bi_tensor_info.hpp>
#include <data/core/bi_types.hpp>
#include <data/core/bi_vlidate.hpp>

#include <common/bi_registers.hpp>

#include <data/core/cpp/bi_cpp_validate.hpp>
#include <data/core/helpers/bi_auto_configuration.hpp>
#include <data/core/helpers/bi_window_helpers.hpp>
#include <data/core/neon/wrapper/wrapper.hpp>
#include <cpu/kernels/select/list.hpp>
#include <arm_neon.h>
#include <map>
#include <string>

namespace BatmanInfer {
    namespace {
        struct SelectKernelSelectorData {
            BIDataType dt;
            bool is_same_rank;
        };

        using SelectorPtr = std::add_pointer<bool(const SelectKernelSelectorData &data)>::type;
        using KernelPtr =
        std::add_pointer<void(const BIITensor *, const BIITensor *, const BIITensor *, BIITensor *,
                              const BIWindow &)>::type;

        struct SelectKernelSelector {
            const char *name;
            const SelectorPtr is_selected;
            KernelPtr ukernel;
        };

        static const SelectKernelSelector available_kernels[] = {
            {
                "neon_s8_same_rank",
                [](const SelectKernelSelectorData &data) {
                    return data.dt == BIDataType::S8 && data.is_same_rank == true;
                },
                REGISTER_INTEGER_NEON(cpu::neon_s8_select_same_rank)
            },
            {
                "neon_s16_same_rank",
                [](const SelectKernelSelectorData &data) {
                    return data.dt == BIDataType::S16 && data.is_same_rank == true;
                },
                REGISTER_INTEGER_NEON(cpu::neon_s16_select_same_rank)
            },
            {
                "neon_s32_same_rank",
                [](const SelectKernelSelectorData &data) {
                    return data.dt == BIDataType::S32 && data.is_same_rank == true;
                },
                REGISTER_INTEGER_NEON(cpu::neon_s32_select_same_rank)
            },
            {
                "neon_u8_same_rank",
                [](const SelectKernelSelectorData &data) {
                    return data.dt == BIDataType::U8 && data.is_same_rank == true;
                },
                REGISTER_INTEGER_NEON(cpu::neon_u8_select_same_rank)
            },
            {
                "neon_u16_same_rank",
                [](const SelectKernelSelectorData &data) {
                    return data.dt == BIDataType::U16 && data.is_same_rank == true;
                },
                REGISTER_INTEGER_NEON(cpu::neon_u16_select_same_rank)
            },
            {
                "neon_u32_same_rank",
                [](const SelectKernelSelectorData &data) {
                    return data.dt == BIDataType::U32 && data.is_same_rank == true;
                },
                REGISTER_INTEGER_NEON(cpu::neon_u32_select_same_rank)
            },
            {
                "neon_s8_not_same_rank",
                [](const SelectKernelSelectorData &data) {
                    return data.dt == BIDataType::S8 && data.is_same_rank == false;
                },
                REGISTER_INTEGER_NEON(cpu::neon_s8_select_not_same_rank)
            },
            {
                "neon_s16_not_same_rank",
                [](const SelectKernelSelectorData &data) {
                    return data.dt == BIDataType::S16 && data.is_same_rank == false;
                },
                REGISTER_INTEGER_NEON(cpu::neon_s16_select_not_same_rank)
            },
            {
                "neon_s32_not_same_rank",
                [](const SelectKernelSelectorData &data) {
                    return data.dt == BIDataType::S32 && data.is_same_rank == false;
                },
                REGISTER_INTEGER_NEON(cpu::neon_s32_select_not_same_rank)
            },
            {
                "neon_u8_not_same_rank",
                [](const SelectKernelSelectorData &data) {
                    return data.dt == BIDataType::U8 && data.is_same_rank == false;
                },
                REGISTER_INTEGER_NEON(cpu::neon_u8_select_not_same_rank)
            },
            {
                "neon_u16_not_same_rank",
                [](const SelectKernelSelectorData &data) {
                    return data.dt == BIDataType::U16 && data.is_same_rank == false;
                },
                REGISTER_INTEGER_NEON(cpu::neon_u16_select_not_same_rank)
            },
            {
                "neon_u32_not_same_rank",
                [](const SelectKernelSelectorData &data) {
                    return data.dt == BIDataType::U32 && data.is_same_rank == false;
                },
                REGISTER_INTEGER_NEON(cpu::neon_u32_select_not_same_rank)
            },
            {
                "neon_f16_same_rank",
                [](const SelectKernelSelectorData &data) {
                    return data.dt == BIDataType::F16 && data.is_same_rank == true;
                },
                REGISTER_FP16_NEON(cpu::neon_f16_select_same_rank)
            },
            {
                "neon_f16_not_same_rank",
                [](const SelectKernelSelectorData &data) {
                    return data.dt == BIDataType::F16 && data.is_same_rank == false;
                },
                REGISTER_FP16_NEON(cpu::neon_f16_select_not_same_rank)
            },
            {
                "neon_f32_same_rank",
                [](const SelectKernelSelectorData &data) {
                    return data.dt == BIDataType::F32 && data.is_same_rank == true;
                },
                REGISTER_FP32_NEON(cpu::neon_f32_select_same_rank)
            },
            {
                "neon_f32_not_same_rank",
                [](const SelectKernelSelectorData &data) {
                    return data.dt == BIDataType::F32 && data.is_same_rank == false;
                },
                REGISTER_FP32_NEON(cpu::neon_f32_select_not_same_rank)

            },
        };

        const SelectKernelSelector *get_implementation(const SelectKernelSelectorData &data) {
            for (const auto &uk: available_kernels) {
                if (uk.is_selected(data)) {
                    return &uk;
                }
            }
            return nullptr;
        }
    } // namespace

    BINESelectKernel::BINESelectKernel()
        : /*_function(nullptr), */ _c(nullptr), _x(nullptr), _y(nullptr), _output(nullptr), _has_same_rank(false) {
    }

    void BINESelectKernel::configure(const BIITensor *c, const BIITensor *x, const BIITensor *y, BIITensor *output) {
        BI_COMPUTE_ERROR_ON_NULLPTR(c, x, y, output);

        // Auto initialize output if not initialized
        auto_init_if_empty(*output->info(), x->info()->tensor_shape(), 1, x->info()->data_type());
        BI_COMPUTE_ERROR_THROW_ON(validate(c->info(), x->info(), y->info(), output->info()));

        _c = c;
        _x = x;
        _y = y;
        _output = output;
        _has_same_rank = (c->info()->tensor_shape().num_dimensions() == x->info()->tensor_shape().num_dimensions());

        BIWindow win = calculate_max_window(*x->info());
        BIINEKernel::configure(win);
    }

    BIStatus
    BINESelectKernel::validate(const BIITensorInfo *c, const BIITensorInfo *x, const BIITensorInfo *y,
                               const BIITensorInfo *output) {
        BI_COMPUTE_RETURN_ERROR_ON_NULLPTR(c, x, y);
        BI_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(x);
        BI_COMPUTE_RETURN_ERROR_ON(x->data_type() == BIDataType::UNKNOWN);
        BI_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(x, y);
        BI_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(x, y);
        BI_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(c, 1, BIDataType::U8);

        const bool is_same_rank = (c->tensor_shape().num_dimensions() == x->tensor_shape().num_dimensions());
        BI_COMPUTE_RETURN_ERROR_ON(is_same_rank && (x->tensor_shape() != c->tensor_shape()));
        BI_COMPUTE_RETURN_ERROR_ON(!is_same_rank &&
            ((c->tensor_shape().num_dimensions() > 1) ||
                (c->tensor_shape().x() != x->tensor_shape()[
                    x->tensor_shape().num_dimensions() - 1])));

        if (output != nullptr && output->total_size() != 0) {
            BI_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(x, output);
            BI_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(x, output);
        }

        return BIStatus{};
    }

    void BINESelectKernel::run(const BIWindow &window, const ThreadInfo &info) {
        BI_COMPUTE_UNUSED(info);
        BI_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
        BI_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(BIINEKernel::window(), window);
        BI_COMPUTE_ERROR_ON(_output == nullptr);
        BI_COMPUTE_ERROR_ON(_output->info() == nullptr);

        const auto *uk = get_implementation(SelectKernelSelectorData{_output->info()->data_type(), _has_same_rank});
        BI_COMPUTE_ERROR_ON(uk == nullptr);
        BI_COMPUTE_ERROR_ON(uk->ukernel == nullptr);
        uk->ukernel(_c, _x, _y, _output, window);
    }
}
