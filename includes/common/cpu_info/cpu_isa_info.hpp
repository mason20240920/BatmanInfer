//
// Created by Mason on 2025/1/3.
//

#ifndef BATMANINFER_CPU_ISA_INFO_HPP
#define BATMANINFER_CPU_ISA_INFO_HPP

#include <cstdint>

namespace BatmanInfer {
    namespace cpu_info {
        /**
         * @brief CPU ISA（指令集架构）信息
         *        包含与 Arm 架构相关的 ISA 信息
         */
        struct CpuIsaInfo {
            /**
             * @brief SIMD扩展支持
             */
            bool neon{false};
            bool sve{false};
            bool sve2{false};
            bool sme{false};
            bool sme2{false};

            /**
             * @brief 数据类型扩展支持
             */
            bool fp16{false};
            bool bf16{false};
            bool svebf16{false};

            /**
             * @brief 指令支持
             */
            bool dot{false};
            bool i8mm{false};
            bool svei8mm{false};
            bool svef32mm{false};
        };

        /**
         * @brief 通过系统信息识别与ISA相关的信息。
         *
         * @param hwcaps HWCAPS feature information
         * @param hwcaps2 HWCAPS2 feature information
         * @param midr MIDR value
         * @return
         */
        CpuIsaInfo init_cpu_isa_from_hwcaps(uint32_t hwcaps, uint32_t hwcaps2, uint32_t midr);

        /**
         * @brief 通过寄存器信息识别 ISA（指令集架构）相关信息
         * @param isar0 指令集属性寄存器 0 的值 (ID_AA64ISAR0_EL1)
         * @param isar1 指令集属性寄存器 1 的值 (ID_AA64ISAR1_EL1)
         * @param pfr0  处理器特性寄存器 0 的值 (ID_AA64PFR0_EL1)
         * @param pfr1  处理器特性寄存器 1 的值 (ID_AA64PFR1_EL1)
         * @param svefr0 SVE 特性 ID 寄存器 0 的值 (ID_AA64ZFR0_EL1)
         * @param midr  主 ID 寄存器的值 (MIDR)
         * @return
         */
        CpuIsaInfo
        init_cpu_isa_from_regs(uint64_t isar0, uint64_t isar1, uint64_t pfr0, uint64_t pfr1, uint64_t svefr0, uint64_t midr);
    }
}

#endif //BATMANINFER_CPU_ISA_INFO_HPP
