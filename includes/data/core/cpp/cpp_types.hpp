//
// Created by Mason on 2025/1/3.
//

#ifndef BATMANINFER_CPP_TYPES_HPP
#define BATMANINFER_CPP_TYPES_HPP

#include <memory>

namespace BatmanInfer {
    namespace cpu_info {
        struct CpuIsaInfo;
    }

#define BI_COMPUTE_CPU_MODEL_LIST \
    X(GENERIC)                     \
    X(GENERIC_FP16)                \
    X(GENERIC_FP16_DOT)            \
    X(A53)                         \
    X(A55r0)                       \
    X(A55r1)                       \
    X(A35)                         \
    X(A73)                         \
    X(A76)                         \
    X(A510)                        \
    X(X1)                          \
    X(V1)                          \
    X(A64FX)                       \
    X(N1)

    /**
     * @brief CPU 模型类型
     *
     * @note 我们只需要检测那些我们为其提供微架构特定代码的 CPU。
     * @note 架构特性是通过 HWCAP（硬件能力标识）检测的。
     */
    enum class CPUModel {
#define X(model) model,
        BI_COMPUTE_CPU_MODEL_LIST
#undef X
    };

    class CPUInfo final {
    protected:
        CPUInfo();
        ~CPUInfo();

    public:
        /**
         * @brief 访问 KernelLibrary 单例。
         *        此方法已被弃用，并将在未来的版本中删除
         * @return
         */
        static CPUInfo &get();

        CPUInfo(CPUInfo const &) = delete;
        CPUInfo(CPUInfo &&) = delete;
        CPUInfo &operator=(CPUInfo const &) = delete;
        CPUInfo &operator=(CPUInfo &&) = delete;

        /** Checks if the cpu model supports fp16.
     *
     * @return true if the cpu supports fp16, false otherwise
     */
        bool has_fp16() const;
        /** Checks if the cpu model supports bf16.
         *
         * @return true if the cpu supports bf16, false otherwise
         */
        bool has_bf16() const;
        /** Checks if the cpu model supports bf16.
         *
         * @return true if the cpu supports bf16, false otherwise
         */
        bool has_svebf16() const;
        /** Checks if the cpu model supports dot product.
         *
         * @return true if the cpu supports dot product, false otherwise
         */
        bool has_dotprod() const;
        /** Checks if the cpu model supports floating-point matrix multiplication.
         *
         * @return true if the cpu supports floating-point matrix multiplication, false otherwise
         */
        bool has_svef32mm() const;
        /** Checks if the cpu model supports integer matrix multiplication.
         *
         * @return true if the cpu supports integer matrix multiplication, false otherwise
         */
        bool has_i8mm() const;
        /** Checks if the cpu model supports integer matrix multiplication.
         *
         * @return true if the cpu supports integer matrix multiplication, false otherwise
         */
        bool has_svei8mm() const;
        /** Checks if the cpu model supports sve.
         *
         * @return true if the cpu supports sve, false otherwise
         */
        bool has_sve() const;
        /** Checks if the cpu model supports sve2.
         *
         * @return true if the cpu supports sve2, false otherwise
         */
        bool has_sve2() const;
        /** Checks if the cpu model supports sme.
         *
         * @return true if the cpu supports sme, false otherwise
         */
        bool has_sme() const;
        /** Checks if the cpu model supports sme2.
         *
         * @return true if the cpu supports sme2, false otherwise
         */
        bool has_sme2() const;
        /** Gets the cpu model for a given cpuid.
         *
         * @param[in] cpuid the id of the cpu core to be retrieved,
         *
         * @return the @ref CPUModel of the cpuid queiried.
         */
        CPUModel get_cpu_model(unsigned int cpuid) const;
        /** Gets the current thread's cpu model
         *
         * @return Current thread's @ref CPUModel
         */
        CPUModel get_cpu_model() const;
        /** Gets the current cpu's ISA information
         *
         * @return Current cpu's ISA information
         */
        cpu_info::CpuIsaInfo get_isa() const;
        /** Gets the L1 cache size
         *
         * @return the size of the L1 cache
         */
        unsigned int get_L1_cache_size() const;
        /** Gets the L2 cache size
         *
         * @return the size of the L1 cache
         */
        unsigned int get_L2_cache_size() const;
        /** Return the maximum number of CPUs present
         *
         * @return Number of CPUs
         */
        unsigned int get_cpu_num() const;
        /** Return the maximum number of CPUs present excluding the little cores
         * in case of an Android device
         *
         * @return Number of CPUs excluding little
         */
        unsigned int get_cpu_num_excluding_little() const;
        /** Return the vector length in bytes for sme2
         *
         * @return Vector length if sme2 is enabled, otherwise returns 0.
         */
        uint64_t get_sme2_vector_length_in_bytes() const;
        /** Return the vector length in bits for sme2
         *
         * @return Vector length if sme2 is enabled, otherwise returns 0.
         */
        uint64_t get_sme2_vector_length_in_bits() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> _impl;
    };

    struct ThreadInfo
    {
        int thread_id{0};
        int num_threads{1};
        const CPUInfo *cpu_info{nullptr};
    };
}

#endif //BATMANINFER_CPP_TYPES_HPP
