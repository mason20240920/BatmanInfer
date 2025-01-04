//
// Created by Mason on 2025/1/3.
//

#ifndef BATMANINFER_CPU_INFO_HPP
#define BATMANINFER_CPU_INFO_HPP

#include <common/cpu_info/cpu_isa_info.hpp>
#include <common/cpu_info/cpu_model.hpp>

#include <string>
#include <vector>

namespace BatmanInfer {
    namespace cpu_info {
        /**
         * @brief  包含 CPU 相关信息的聚合类
         *         - CPU 的数量
         *         - 每个 CPU 的型号
         *         - 与指令集架构（ISA）相关的信息
         *         - 以及其他更多信息
         *
         * @note 我们可以安全地假设不同核心簇之间的指令集架构（ISA）是相同的
         *
         */
        class CpuInfo {
        public:
            /** Default constructor */
            CpuInfo() = default;

            /** Construct a new Cpu Info object
             *
             * @param[in] isa  指令集属性信息
             * @param[in] cpus CPU模型信息
             */
            CpuInfo(CpuIsaInfo isa, std::vector<CpuModel> cpus);

            /** 系统相关信息创建CPU
             *
             * @return CpuInfo A populated CpuInfo structure
             */
            static CpuInfo build();

        public:
            bool has_neon() const {
                return _isa.neon;
            }

            bool has_sve() const {
                return _isa.sve;
            }

            bool has_sve2() const {
                return _isa.sve2;
            }

            bool has_sme() const {
                return _isa.sme;
            }

            bool has_sme2() const {
                return _isa.sme2;
            }

            bool has_fp16() const {
                return _isa.fp16;
            }

            bool has_bf16() const {
                return _isa.bf16;
            }

            bool has_svebf16() const {
                return _isa.svebf16;
            }

            bool has_dotprod() const {
                return _isa.dot;
            }

            bool has_i8mm() const {
                return _isa.i8mm;
            }

            bool has_svei8mm() const {
                return _isa.svei8mm;
            }

            bool has_svef32mm() const {
                return _isa.svef32mm;
            }

            const CpuIsaInfo &isa() const {
                return _isa;
            }

            const std::vector<CpuModel> &cpus() const {
                return _cpus;
            }

            CpuModel cpu_model(uint32_t cpuid) const;

            CpuModel cpu_model() const;

            uint32_t num_cpus() const;

            uint32_t not_little_num_cpus() const;

        private:
            CpuIsaInfo            _isa{};
            std::vector<CpuModel> _cpus{};
        };

        /** 一些系统同时具有大核和小核，该函数计算系统中完全相同核心的最小数量。
         *  为了最大化性能，库会尝试使用系统上可用的大核线程数量并发处理工作负载。
         *
         * @return 最小的公共核心数量。
         */
        uint32_t num_threads_hint();
    }
}

#endif //BATMANINFER_CPU_INFO_HPP
