//
// Created by Mason on 2025/1/3.
//

#include <data/core/cpp/cpp_types.hpp>
#include "common/cpu_info/cpu_isa_info.hpp"
#include <common/cpu_info/cpu_info.hpp>

#include <data/core/neon/kernels/arm_gemm/utils.hpp>

namespace BatmanInfer {
    struct CPUInfo::Impl {
        cpu_info::CpuInfo info{};
        unsigned int     L1_cache_size = 32768;
        unsigned int     L2_cache_size = 262144;
    };

    CPUInfo &CPUInfo::get() {
        static CPUInfo _cpu_info;
        return _cpu_info;
    }

    CPUInfo::CPUInfo() : _impl(std::make_unique<Impl>())
    {
        _impl->info = cpu_info::CpuInfo::build();
    }

    unsigned int CPUInfo::get_cpu_num() const
    {
        return _impl->info.num_cpus();
    }

    bool CPUInfo::has_fp16() const
    {
        return _impl->info.has_fp16();
    }

    bool CPUInfo::has_bf16() const
    {
        return _impl->info.has_bf16();
    }

    bool CPUInfo::has_svebf16() const
    {
        return _impl->info.has_svebf16();
    }

    bool CPUInfo::has_dotprod() const
    {
        return _impl->info.has_dotprod();
    }

    bool CPUInfo::has_svef32mm() const
    {
        return _impl->info.has_svef32mm();
    }

    bool CPUInfo::has_i8mm() const
    {
        return _impl->info.has_i8mm();
    }

    bool CPUInfo::has_svei8mm() const
    {
        return _impl->info.has_svei8mm();
    }

    bool CPUInfo::has_sve() const
    {
        return _impl->info.has_sve();
    }

    bool CPUInfo::has_sve2() const
    {
        return _impl->info.has_sve2();
    }

    bool CPUInfo::has_sme() const
    {
        return _impl->info.has_sme();
    }

    CPUInfo::~CPUInfo() = default;

    bool CPUInfo::has_sme2() const
    {
        return _impl->info.has_sme2();
    }

    CPUModel CPUInfo::get_cpu_model() const
    {
        return _impl->info.cpu_model();
    }

    CPUModel CPUInfo::get_cpu_model(unsigned int cpuid) const
    {
        return _impl->info.cpu_model(cpuid);
    }

    cpu_info::CpuIsaInfo CPUInfo::get_isa() const
    {
        return _impl->info.isa();
    }

    unsigned int CPUInfo::get_L1_cache_size() const
    {
        return _impl->L1_cache_size;
    }

    unsigned int CPUInfo::get_L2_cache_size() const
    {
        return _impl->L2_cache_size;
    }

    uint64_t CPUInfo::get_sme2_vector_length_in_bytes() const
    {
#ifdef ARM_COMPUTE_ENABLE_SME2
        if (this->has_sme2())
        return arm_gemm::utils::sme::get_vector_length<int8_t>();
    else
        return 0;
#else  // ARM_COMPUTE_ENABLE_SME2
        return 0;
#endif // ARM_COMPUTE_ENABLE_SME2
    }

    uint64_t CPUInfo::get_sme2_vector_length_in_bits() const
    {
        return get_sme2_vector_length_in_bytes() * 8;
    }

    unsigned int CPUInfo::get_cpu_num_excluding_little() const {
#if defined(__ANDROID__)
        return _impl->info.not_little_num_cpus();
#else  /* defined(__ANDROID__) */
        return get_cpu_num();
#endif /* defined(__ANDROID__) */
    }
}