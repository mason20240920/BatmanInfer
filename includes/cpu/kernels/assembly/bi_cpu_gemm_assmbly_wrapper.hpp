//
// Created by Mason on 2025/1/5.
//

#ifndef BATMANINFER_BI_CPU_GEMM_ASSEMBLY_WRAPPER_HPP
#define BATMANINFER_BI_CPU_GEMM_ASSEMBLY_WRAPPER_HPP

#include <data/core/bi_error.h>
#include <data/core/experimental/types.hpp>
#include <data/core/bi_utils.hpp>
#include <data/core/bi_vlidate.hpp>

#include <data/core/neon/bi_i_ne_kernel.hpp>
#include <cpu/kernels/assembly/batman_gemm_compute_iface.hpp>

#include <cpu/kernels/assembly/bi_gemm_arrays.hpp>
#include <cpu/kernels/assembly/bi_gemm_common.hpp>

namespace BatmanInfer {
    class BIITensor;

    namespace cpu {
        namespace kernel {
            /** 该类是用于汇编内核的封装器。
            *
            * 一些内核是用汇编编写的，并针对特定的 CPU（如 A53 或 A55）进行了高度优化。
            * 该类作为这些汇编内核的封装器使用。BatmanInfer Library 会创建一个
            * BICpuGemmAssemblyWrapperKernel 实例以及其他辅助数据结构，以在 BINEFunctions 的上下文中
            * 执行单个汇编内核。
            *
            * 模板参数 T 是用汇编实现的实际内核的类型，其类型为：
            *         template<typename To, typename Tr> class GemmCommon
            *
            */
            template<typename TypeInput, typename TypeWeight, typename TypeOutput>
            class BICpuGemmAssemblyWrapperKernel final : public BIINEKernel {
            public:
                /**
                 * 构造函数
                 * @return
                 */
                BICpuGemmAssemblyWrapperKernel() : _kernel(nullptr), _name("BICpuGemmAssemblyWrapperKernel") {
                }

                BICpuGemmAssemblyWrapperKernel(BICpuGemmAssemblyWrapperKernel
                                               &) = delete;

                BICpuGemmAssemblyWrapperKernel(BICpuGemmAssemblyWrapperKernel
                                               &&) = default;

                BICpuGemmAssemblyWrapperKernel &operator=(BICpuGemmAssemblyWrapperKernel &) = delete;

                const char *name() const override {
                    return _name.c_str();
                }

                void run(const BatmanInfer::BIWindow &window, const BatmanInfer::ThreadInfo &info) override {
                    BI_COMPUTE_ERROR_ON_NULLPTR((reinterpret_cast<void *>(_kernel)));
                    BI_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);

                    auto win = BatmanGemm::to_ndcoord(window);

                    BatmanGemm::ndcoord_t thread_locator{};
                    _kernel->execute(win, thread_locator, info.thread_id);
                }

                void run_nd(const BatmanInfer::BIWindow &window, const BatmanInfer::ThreadInfo &info,
                            const BatmanInfer::BIWindow &thread_locator) override {
                    BI_COMPUTE_ERROR_ON_NULLPTR((reinterpret_cast<void *>(_kernel)));
                    BI_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);

                    //convert between arm_compute and arm_gemm types
                    auto ndc_win = BatmanGemm::to_ndcoord(window);
                    auto ndc_tlc = BatmanGemm::to_ndcoord(thread_locator);

                    _kernel->execute(ndc_win, ndc_tlc, info.thread_id);
                }

                void run_op(BIITensorPack &tensors,
                            const BIWindow &window,
                            const ThreadInfo &info) override {
                    BI_COMPUTE_ERROR_ON_NULLPTR((reinterpret_cast<void *>(_kernel)));
                    BI_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);

                    const auto *Aptr = reinterpret_cast<const TypeInput *>(tensors.get_tensor(ACL_SRC_0)->buffer());
                    const auto *Bptr = reinterpret_cast<const TypeWeight *>(tensors.get_tensor(ACL_SRC_1)->buffer());
                    const auto *bias = reinterpret_cast<const TypeOutput *>(tensors.get_tensor(ACL_SRC_2)->buffer());
                    auto *Cptr = reinterpret_cast<TypeOutput *>(tensors.get_tensor(ACL_DST)->buffer());
                    // dynamic set M size
                    const auto Msize = tensors.get_tensor(ACL_SRC_0)->info()->tensor_shape().y();

                    BI_COMPUTE_ERROR_ON_NULLPTR(Aptr, Cptr);

                    // We make a copy of the original gemm arrays and then update the
                    // source, bias, and destination pointers with the packed values.
                    BatmanGemm::BIGemmArrays<TypeInput, TypeWeight, TypeOutput> ga = _kernel->get_gemm_arrays();

                    ga._Aptr = Aptr;
                    ga._Bptr = Bptr;
                    ga._bias = bias;
                    ga._Cptr = Cptr;

                    auto win = BatmanGemm::to_ndcoord(window);

                    BatmanGemm::ndcoord_t thread_locator{};

                    _kernel->set_dynamic_M_size(Msize);
                    _kernel->execute_stateless(win, thread_locator, info.thread_id, ga);
                }

                /**
                 * 配置内核的窗口
                 * @param win 用于执行内核的窗口区域
                 */
                void configure_window(const BIWindow &win) {
                    BIINEKernel::configure(win);
                }

                /**
                 * 配置内核的输入和输出
                 * @param kernel
                 * @param kernel_name_tag
                 */
                void configure(BatmanGemm::BIGemmCommon<TypeInput, TypeWeight, TypeOutput> *kernel,
                               std::string kernel_name_tag) {
                    BI_COMPUTE_ERROR_ON_NULLPTR((reinterpret_cast<void *>(kernel)));
                    _kernel = kernel;

                    BIWindow win = to_window(kernel->get_window_size());

                    BIINEKernel::configure(win);

                    if (!kernel_name_tag.empty()) {
                        _name += "/" + kernel_name_tag;
                    }
                }


                size_t get_mws(const BatmanInfer::CPUInfo &platform, size_t thread_count) const override {
                    BI_COMPUTE_UNUSED(thread_count);
                    BI_COMPUTE_UNUSED(platform);

                    return BIICPPKernel::default_mws;
                }


            private:
                BatmanGemm::BIGemmCommon<TypeInput, TypeWeight, TypeOutput> *_kernel;
                std::string _name;
            };
        } // namespace kernel
    }
}

#endif //BATMANINFER_BI_CPU_GEMM_ASSEMBLY_WRAPPER_HPP
