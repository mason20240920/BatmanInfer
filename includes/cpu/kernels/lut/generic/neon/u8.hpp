//
// Created by Mason on 2025/1/10.
//

#ifndef BATMANINFER_CPU_KERNELS_LUT_U8_HPP
#define BATMANINFER_CPU_KERNELS_LUT_U8_HPP

#include <cstddef>
#include <cstdint>

namespace BatmanInfer {
    namespace cpu {
#ifdef __aarch64__

        void lut_u8_neon(
                const uint8_t *table,
                size_t num_strings,
                size_t string_length,
                const uint8_t *const *input,
                uint8_t *const *output);

#endif
    }
}

#endif //BATMANINFER_CPU_KERNELS_LUT_U8_HPP
