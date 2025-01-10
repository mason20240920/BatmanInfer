//
// Created by Mason on 2025/1/10.
//

#include <data/core/bi_helpers.hpp>

#include <function_info/bi_activationLayerInfo.h>

#include <cpu/kernels/lut/list.hpp>

namespace BatmanInfer {
    namespace cpu {
#ifdef __aarch64__

        void neon_q8_activation_lut(const BIITensor *src,
                                    BIITensor *dst,
                                    const BIActivationLayerInfo &act_info,
                                    const BIWindow &window) {
            BI_COMPUTE_ERROR_ON( // LUT 对 ReLU 没有性能提升，因为它只是一个单一的 max() 操作。
                    (src->info()->data_type() != BIDataType::QASYMM8 &&
                     src->info()->data_type() != BIDataType::QASYMM8_SIGNED) ||
                    act_info.activation() == BIActivationLayerInfo::ActivationFunction::RELU);
            const auto window_end_x = window.x().end();
            BIWindow win_collapsed = window.collapse_if_possible(window, BIWindow::DimZ);
            win_collapsed.set(BIWindow::DimX, BIWindow::BIDimension(0, 1, 1));
            BIIterator input(src, win_collapsed);
            BIIterator output(dst, win_collapsed);
            execute_window_loop(
                    win_collapsed,
                    [&](const BICoordinates &) {
                        const auto input_ptr = reinterpret_cast<const uint8_t *>(input.ptr());
                        auto output_ptr = reinterpret_cast<uint8_t *>(output.ptr());
                        lut_u8_neon(act_info.lut().data(), 1u, window_end_x, &input_ptr, &output_ptr);
                    },
                    input, output);
        }

#endif
    }
}