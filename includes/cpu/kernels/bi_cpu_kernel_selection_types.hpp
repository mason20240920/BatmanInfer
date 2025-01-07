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
            /**
             * @brief 数据类型（BIDataType）和CPU指令集信息（cpu_info::CpuIsaInfo）
             *        用于选择基于数据类型和CPU指令集的操作
             */
            struct BIDataTypeISASelectorData {
                BIDataType           dt;
                cpu_info::CpuIsaInfo isa;
            };

            /**
             * @brief 用于结合数据类型和布局的信息选择操作
             */
            struct BIDataTypeDataLayoutISASelectorData {
                BIDataType                 dt;
                BIDataLayout               dl;
                const cpu_info::CpuIsaInfo &isa;
            };

            /**
             * @brief 数据类型转换操作的选择
             */
            struct BICastDataTypeISASelectorData {
                BIDataType                 src_dt;
                BIDataType                 dst_dt;
                const cpu_info::CpuIsaInfo &isa;
            };

            /**
             * @brief 专用于池化操作的选择
             */
            struct PoolDataTypeISASelectorData {
                BIDataType           dt;
                BIDataLayout         dl;
                int                  pool_stride_x;
                Size2D               pool_size;
                cpu_info::CpuIsaInfo isa;
            };

            /**
             * @brief 选择逐元素操作的实现
             */
            struct ElementwiseDataTypeISASelectorData {
                BIDataType           dt;
                cpu_info::CpuIsaInfo isa;
                int                  op;
            };

            /**
             * @brief 专用于深度卷积操作的选择
             */
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
                BIDataType            dt;
                cpu_info::CpuIsaInfo  isa;
                BIInterpolationPolicy interpolation_policy;
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

            // Selector pointer types
            /**
             * @brief  类型特征（type trait）
             */
            using DataTypeISASelectorPtr = std::add_pointer<bool(const BIDataTypeISASelectorData &data)>::type;
            using DataTypeDataLayoutSelectorPtr = std::add_pointer<bool(
                    const BIDataTypeDataLayoutISASelectorData &data)>::type;
            using PoolDataTypeISASelectorPtr = std::add_pointer<bool(const PoolDataTypeISASelectorData &data)>::type;
            using ElementwiseDataTypeISASelectorPtr = std::add_pointer<bool(
                    const ElementwiseDataTypeISASelectorData &data)>::type;
            using DepthwiseConv2dNativeDataTypeISASelectorPtr =
                    std::add_pointer<bool(const DepthwiseConv2dNativeDataTypeISASelectorData &data)>::type;
            using CastDataTypeISASelectorDataPtr = std::add_pointer<bool(
                    const BICastDataTypeISASelectorData &data)>::type;
            using ActivationDataTypeISASelectorDataPtr =
                    std::add_pointer<bool(const ActivationDataTypeISASelectorData &data)>::type;
            using CpuAddKernelDataTypeISASelectorDataPtr =
                    std::add_pointer<bool(const CpuAddKernelDataTypeISASelectorData &data)>::type;
            using ScaleKernelDataTypeISASelectorDataPtr =
                    std::add_pointer<bool(const ScaleKernelDataTypeISASelectorData &data)>::type;
            using SoftmaxKernelDataTypeISASelectorDataPtr =
                    std::add_pointer<bool(const SoftmaxKernelDataTypeISASelectorData &data)>::type;
            using ScatterKernelDataTypeISASelectorDataPtr =
                    std::add_pointer<bool(const ScatterKernelDataTypeISASelectorData &data)>::type;
        } // kernels
    }
}

#endif //BATMANINFER_BI_CPU_KERNEL_SELECTION_TYPES_HPP
