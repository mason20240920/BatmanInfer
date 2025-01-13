//
// Created by Mason on 2025/1/13.
//

#ifndef NO_MULTI_THREADING

#include <mutex>

#endif

#include <cstdint>

#include <cpu/kernels/assembly/bi_arm_gemm.hpp>
#include <data/core/neon/kernels/arm_gemm/utils.hpp>
#include <data/core/neon/kernels/arm_gemm/kernel_weight_format.hpp>

namespace BatmanGemm {

#ifndef NO_MULTI_THREADING
    std::mutex report_mutex;
#endif

    WeightFormat get_weight_format(const KernelWeightFormat kwf, size_t element_size) {
        if (kwf == KernelWeightFormat::NON_FIXED) {
            return WeightFormat::UNSPECIFIED;
        }

        uint32_t kwf_i = static_cast<uint32_t>(kwf);
        uint32_t wf_i = 0;

        const auto block_bytes = (kwf_i >> 8) & 0xf;
        const auto vector_count = (kwf_i >> 12) & 0xf;

        uint32_t vector_bytes;

        // For fast mode BF16 kernels set the appropriate bit and override element size to 2.
        if (kwf_i & 0x10) {
            element_size = 2;
            wf_i |= 0x10;
        }

#ifdef BI_COMPUTE_ENABLE_SVE
        // Get total bytes in vector output
    if (kwf_i & 0x1) {
        vector_bytes = vector_count * get_vector_length<uint8_t>();
    } else {
#else
        if (1) {
#endif
            vector_bytes = vector_count * 16;
        }

        auto input_blocking = block_bytes / element_size;
        auto output_blocking = vector_bytes / block_bytes;

        wf_i |= (input_blocking << 20);
        wf_i |= (output_blocking << 8);

        return static_cast<WeightFormat>(wf_i);
    }

} // namespace BatmanGemm