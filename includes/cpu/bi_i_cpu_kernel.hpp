//
// Created by Mason on 2025/1/6.
//

#ifndef BATMANINFER_BI_I_CPU_KERNEL_HPP
#define BATMANINFER_BI_I_CPU_KERNEL_HPP

#include <data/core/cpp/bi_i_cpp_kernel.hpp>

#include <cpu/kernels/bi_cpu_kernel_selection_types.hpp>

namespace BatmanInfer {
    namespace cpu {
        enum class BIKernelSelectionType {
            /**
             * @brief 获取针对给定 CPU ISA 的最佳实现，不考虑当前的构建标志
             */
            Preferred,
            /**
             * @brief 获取针对给定 CPU ISA 且受当前构建支持的最佳实现
             */
            Supported
        };

        template<class Derived>
        class BIICpuKernel : public BIICPPKernel {
        public:
            /**
             * @brief 根据特定的选择器和选择策略，挑选最合适的微内核（micro-kernel）实现
             * @tparam SelectorType 模板参数，表示选择器的类型
             * @param selector 是一个选择器对象，包含用于选择微内核的相关信息（如数据类型、指令集等）
             * @param selection_type 枚举类型 KernelSelectionType，用于指定选择策略
             * @return
             */
            template<typename SelectorType>
            static const auto *get_implementation(const SelectorType &selector,
                                                  BIKernelSelectionType selection_type = BIKernelSelectionType::Supported) {
                // Derived::get_available_kernels() 是派生类提供的静态方法，返回一个可用微内核列表
                // decltype 获取返回值的类型，std::remove_reference 去掉可能的引用修饰符，最终得到列表中元素的类型（即 kernel_type）
                using kernel_type = typename std::remove_reference<decltype(Derived::get_available_kernels())>::type::value_type;

                for (const auto &uk: Derived::get_available_kernels()) {
                    if (uk.is_selected(selector) &&
                        (selection_type == BIKernelSelectionType::Preferred || uk.ukernel != nullptr))
                        return &uk;
                }

                return static_cast<kernel_type *>(nullptr);
            }
        };
    }
}

#endif //BATMANINFER_BI_I_CPU_KERNEL_HPP
