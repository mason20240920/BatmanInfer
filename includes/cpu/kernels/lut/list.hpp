//
// Created by Mason on 2025/1/10.
//

#ifndef BATMANINFER_CPU_KERNELS_LIST_HPP
#define BATMANINFER_CPU_KERNELS_LIST_HPP

#include <cpu/kernels/lut/generic/neon/u8.hpp>

namespace BatmanInfer {
    namespace cpu {
#ifdef __aarch64__
#define DECLARE_LUT_U8_KERNEL(func_name)                                                                        \
    void func_name(const uint8_t *table, size_t num_strings, size_t string_length, const uint8_t *const *input, \
                   uint8_t *const *output)

        DECLARE_LUT_U8_KERNEL(lut_u8_neon);

#undef DECLARE_LUT_U8_KERNEL

#define DECLARE_LUT_U16_KERNEL(func_name)                                                                  \
    void func_name(const uint16_t *table, size_t num_strings, size_t string_length, const uint16_t *input, \
                   uint16_t *output)

        DECLARE_LUT_U16_KERNEL(lut_u16_neon);

#undef DECLARE_LUT_U16_KERNEL

#endif // __aarch64__
    }
}

#endif //BATMANINFER_CPU_KERNELS_LIST_HPP
