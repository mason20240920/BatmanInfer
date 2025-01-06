//
// Created by Mason on 2025/1/6.
//

#ifndef BATMANINFER_BI_CPU_KERNEL_SELECTION_TYPES_HPP
#define BATMANINFER_BI_CPU_KERNEL_SELECTION_TYPES_HPP

#include <data/core/bi_types.hpp>

#include <common/cpu_info/cpu_isa_info.hpp>
#include <data/core/cpp/cpp_types.hpp>

namespace BatmanInfer {
    namespace cpu {
        namespace kernels {
            struct BIDataTypeISASelectorData {
                BIDataType           dt;
                cpu_info::CpuIsaInfo isa;
            };

            struct BIDataTypeDataLayoutISASelectorData {
                BIDataType                 dt;
                BIDataLayout               dl;
                const cpu_info::CpuIsaInfo &isa;
            };

            struct BICastDataTypeISASelectorData {
                BIDataType                 src_dt;
                BIDataType                 dst_dt;
                const cpu_info::CpuIsaInfo &isa;
            };

            struct PoolDataTypeISASelectorData {
                BIDataType           dt;
                BIDataLayout         dl;
                int                  pool_stride_x;
                Size2D               pool_size;
                cpu_info::CpuIsaInfo isa;
            };

            struct ElementwiseDataTypeISASelectorData {
                BIDataType           dt;
                cpu_info::CpuIsaInfo isa;
                int                  op;
            };
            struct DepthwiseConv2dNativeDataTypeISASelectorData {
                BIDataType                 weights_dt;
                BIDataType                 source_dt;
                const cpu_info::CpuIsaInfo &isa;
            };

            struct ActivationDataTypeISASelectorData {
                BIDataType                 dt;
                const CPUModel             &cpumodel;
                const cpu_info::CpuIsaInfo &isa;
                const BIActivationFunction f;
            };

            struct CpuAddKernelDataTypeISASelectorData {
                BIDataType           dt;
                cpu_info::CpuIsaInfo isa;
                bool                 can_use_fixedpoint;
                bool                 can_use_sme2_impl;
            };

            struct ScaleKernelDataTypeISASelectorData {
                BIDataType           dt;
                cpu_info::CpuIsaInfo isa;
                InterpolationPolicy  interpolation_policy;
            };

            struct SoftmaxKernelDataTypeISASelectorData {
                BIDataType           dt;
                cpu_info::CpuIsaInfo isa;
                bool                 is_log;
                int                  axis;
                uint64_t             sme2_vector_length;
            };

            struct ScatterKernelDataTypeISASelectorData {
                BIDataType           dt;
                cpu_info::CpuIsaInfo isa;
                unsigned long        sme2_vector_length;
            };
        }
    }
}

#endif //BATMANINFER_BI_CPU_KERNEL_SELECTION_TYPES_HPP
