//
// Created by Mason on 2025/1/3.
//

#ifndef BATMANINFER_UTILS_HPP
#define BATMANINFER_UTILS_HPP

#include <cstddef>
#include <limits>
#include <tuple>

namespace arm_gemm {
    namespace utils {
#ifdef __aarch64__
        namespace sme {
            template <typename T>
            inline uint64_t get_vector_length() {
                uint64_t raw_vector_length;

                __asm __volatile (
                        ".inst 0x04bf5821\n" // RDSVL X1, #1
                        "mov %0, X1\n"
                        : "=r" (raw_vector_length)
                        :
                        : "x1"
                        );

                return raw_vector_length / sizeof(T);
            }
        }
#endif
    }
}

#endif //BATMANINFER_UTILS_HPP
