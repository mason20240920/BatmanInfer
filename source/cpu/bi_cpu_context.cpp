//
// Created by Mason on 2025/1/11.
//
#include <cpu/bi_cpu_context.hpp>

#include <data/core/cpp/cpp_types.hpp>

#include <cpu/bi_cpu_queue.hpp>
#include <cpu/bi_cpu_tensor.hpp>

#include <cstdlib>

#if !defined(__APPLE__) && !defined(__OpenBSD__)
#include <malloc.h>

#if defined(_WIN64)
#define posix_memalign      _aligned_realloc
#define posix_memalign_free _aligned_free
#endif // defined(_WIN64)
#endif // !defined(__APPLE__) && !defined(__OpenBSD__)

#ifndef BARE_METAL

#include <thread>

#endif /* BARE_METAL */

namespace BatmanInfer {
    namespace cpu {
        namespace {
            void *default_allocate(void *user_data,
                                   size_t size) {
                BI_COMPUTE_UNUSED(user_data);
                return ::operator new(size);
            }

            void default_free(void *user_data,
                              void *ptr) {
                BI_COMPUTE_UNUSED(user_data);
                ::operator delete(ptr);
            }

            void *default_aligned_allocate(void *user_data,
                                           size_t size,
                                           size_t alignment) {
                BI_COMPUTE_UNUSED(user_data);
                void *ptr = nullptr;
#if defined(BARE_METAL)
                size_t rem       = size % alignment;
                size_t real_size = (rem) ? (size + alignment - rem) : size;
                ptr              = memalign(alignment, real_size);
#else  /* defined(BARE_METAL) */
                if (posix_memalign(&ptr, alignment, size) != 0) {
                    // posix_memalign returns non-zero on failures, the return values will be
                    // - EINVAL: wrong alignment
                    // - ENOMEM: insufficient memory
                    BI_COMPUTE_LOG_ERROR_ACL("posix_memalign failed, the returned pointer will be invalid");
                }
#endif /* defined(BARE_METAL) */
                return ptr;
            }

            void default_aligned_free(void *user_data,
                                      void *ptr) {
                BI_COMPUTE_UNUSED(user_data);
                free(ptr);
            }

            static BclAllocator default_allocator = {&default_allocate,
                                                     &default_free,
                                                     &default_aligned_allocate,
                                                     &default_aligned_free,
                                                     nullptr};

            BIAllocatorWrapper populate_allocator(BclAllocator *external_allocator) {
                bool is_valid = (external_allocator != nullptr);
                if (is_valid) {
                    is_valid = is_valid && (external_allocator->alloc != nullptr);
                    is_valid = is_valid && (external_allocator->free != nullptr);
                    is_valid = is_valid && (external_allocator->aligned_alloc != nullptr);
                    is_valid = is_valid && (external_allocator->aligned_free != nullptr);
                }
                return is_valid ? BIAllocatorWrapper(*external_allocator) : BIAllocatorWrapper(default_allocator);
            }

            cpu_info::CpuIsaInfo populate_capabilities_flags(BclTargetCapabilities external_caps) {
                cpu_info::CpuIsaInfo isa_caps;

                // Extract SIMD extension
                isa_caps.neon = external_caps & BclCpuCapabilitiesNeon;
                isa_caps.sve = external_caps & BclCpuCapabilitiesSve;
                isa_caps.sve2 = external_caps & BclCpuCapabilitiesSve2;

                // Extract data-type support
                isa_caps.fp16 = external_caps & BclCpuCapabilitiesFp16;
                isa_caps.bf16 = external_caps & BclCpuCapabilitiesBf16;
                isa_caps.svebf16 = isa_caps.bf16;

                // Extract ISA extensions
                isa_caps.dot = external_caps & BclCpuCapabilitiesDot;
                isa_caps.i8mm = external_caps & BclCpuCapabilitiesMmlaInt8;
                isa_caps.svef32mm = external_caps & BclCpuCapabilitiesMmlaFp;

                return isa_caps;
            }

            BICpuCapabilities populate_capabilities(BclTargetCapabilities external_caps,
                                                    int32_t max_threads) {
                BICpuCapabilities caps;

                // 根据系统信息产生配置
                caps.cpu_info = cpu_info::CpuInfo::build();
                if (external_caps != BclCpuCapabilitiesAuto) {
                    cpu_info::CpuIsaInfo isa = populate_capabilities_flags(external_caps);
                    auto cpus = caps.cpu_info.cpus();

                    caps.cpu_info = cpu_info::CpuInfo(isa, cpus);
                }

                // Set max number of threads
#if defined(BARE_METAL)
                BI_COMPUTE_UNUSED(max_threads);
    caps.max_threads = 1;
#else  /* defined(BARE_METAL) */
                caps.max_threads = (max_threads > 0) ? max_threads : std::thread::hardware_concurrency();
#endif /* defined(BARE_METAL) */

                return caps;
            }

        } // namespace

        BICpuContext::BICpuContext(const BclContextOptions *options) : BIIContext(Target::Cpu),
                                                                       _allocator(default_allocator),
                                                                       _caps(populate_capabilities(
                                                                               BclCpuCapabilitiesAuto, -1)) {
            if (options != nullptr) {
                _allocator = populate_allocator(options->allocator);
                _caps = populate_capabilities(options->capabilities, options->max_compute_units);
            }
        }

        const BICpuCapabilities &BICpuContext::capabilities() const {
            return _caps;
        }

        BIAllocatorWrapper &BICpuContext::allocator() {
            return _allocator;
        }

        BIITensorV2 *BICpuContext::create_tensor(const BclTensorDescriptor &desc, bool allocate) {
            BICpuTensor *tensor = new BICpuTensor(this, desc);
            if (tensor != nullptr && allocate)
                tensor->allocate();
            return tensor;
        }

        BIIQueue *BICpuContext::create_queue(const BclQueueOptions *options) {
            return new BICpuQueue(this, options);
        }
    } // namespace cpu
}