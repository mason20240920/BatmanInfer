//
// Created by Mason on 2025/3/26.
//

#pragma once

#include <data/core/cpp/bi_i_cpp_kernel.hpp>
#include <data/core/bi_i_tensor.hpp>
#include <data/core/bi_i_tensor_info.hpp>
#include <data/core/bi_window.hpp>
#include <runtime/bi_i_scheduler.hpp>

#include <data/core/common/bi_core_common_macros.hpp>
#include <cpu/kernels/bi_cpu_kernel_selection_types.hpp>

#include <map>
#include <vector>

namespace BatmanInfer {
    namespace cpu {
        namespace kernels {
            namespace heuristics {
                class BICpuDynamicGemmKernelHeuristics {
                public:
                    /**
                     * @brief Run the micro-kernel
                     *
                     * @param[in] a     Tensor a
                     * @param[in] b     Tensor b
                     * @param[in] c     Tensor c
                     * @param[in] d     Tensor d
                     * @param[in] pack_b Packed tensor b
                     * @param[in] window Window to run the kernel on
                     */
                    using KernelPtr = std::add_pointer_t<void(
                            const BIITensor *,
                            const BIITensor *,
                            const BIITensor *,
                            BIITensor *,
                            BIITensor *,
                            const BIWindow &)>;

                    /**
                     * @brief 打包右操作数
                     *
                     * @param[in] 右操作数  Tensor b
                     * @param[in] bias     Bias data
                     * @param[out] packed_rhs 用于存放压缩右操作数数据的目标缓冲区
                     */
                    using PackRhsPtr = std::add_pointer_t<void(
                            const BIITensor *,
                            const BIITensor *,
                            BIITensor *
                    )>;

                    /**
                     * @brief Size of packed RHS for data of given size
                     *
                     * @param[in] rows    Number of rows
                     * @param[in] columns Number of columns
                     *
                     * @return Size of packed RHS data
                     */
                    using SizeOfPackedRhsPtr = std::add_pointer_t<size_t(
                            const size_t,
                            const size_t)>;

                    /**
                     * @brief 计算窗口大小
                     *
                     * @param[in] dst 目标张量
                     *
                     * @return 微小内核的窗口大小
                     */
                    using GetWindowPtr = std::add_pointer_t<BIWindow(const BIITensorInfo *dst)>;

                    BI_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(BICpuDynamicGemmKernelHeuristics);

                    // 默认构造函数与析构函数
                    BICpuDynamicGemmKernelHeuristics() noexcept {
                    };

                    ~BICpuDynamicGemmKernelHeuristics() = default;

                    /**
                     * @brief 初始化内核的输入和输出
                     * @param a 第一个输入的张量信息(矩阵A或向量A). 数据类型支持: F32
                     * @param b 第二个输入的张量信息(矩阵B), 数据类型: @p a
                     * @param c 第三个输入的矩阵信息(矩阵C), 能够变成nullptr
                     * @param d 输出矩阵
                     * @param alpha 矩乘的权重
                     * @param beta 矩阵c的权重
                     * @param gemm_info 矩阵A或矩阵B是否已经被reshaped且矩阵B是否已经reshape
                     */
                    BICpuDynamicGemmKernelHeuristics(const BIITensorInfo *a,
                                                     const BIITensorInfo *b,
                                                     const BIITensorInfo *c,
                                                     BIITensorInfo *d,
                                                     float alpha,
                                                     float beta,
                                                     const GEMMInfo &gemm_info = GEMMInfo());

                    /**
                     * @brief 返回最小的workload大小
                     * @return 请求配置的最小工作负载(size_t)
                     */
                    size_t mws() const;

                    /**
                     * @brief 准备微核运行
                     *
                     * 在这里可以进行的一项操作的示例是 b-张量打包。
                     *
                     * @param tensors 运行中会使用的张量
                     * @param run_packing b张量是否需要打包
                     * @param pack_b_tensor_offset tensors 参数中 pack_rhs 张量的偏移量。
                     */
                    void prepare(BIITensorPack &tensors,
                                 bool run_packing,
                                 const int pack_b_tensor_offset);

                    /**
                     * @brief 返回内核去运行
                     * @return 函数指针: 指向被选中的内核
                     */
                    KernelPtr kernel() const;

                    /**
                     * @brief 为内核返回pack_rhs()函数
                     * @return 指针: 指向打包右操作数的指针
                     */
                    PackRhsPtr pack_rhs() const;

                    /**
                     * @brief kernel函数: 返回size_of_packed_rhs()
                     * @return 返回函数指针
                     */
                    SizeOfPackedRhsPtr size_of_packed_rhs() const;

                    /**
                     * @brief 内核返回获取get_window()函数
                     * @return 函数指针: get_window()
                     */
                    GetWindowPtr get_window() const;

                    /**
                     * @brief 返回选中内核的名称
                     * @return 选中内核的名称
                     */
                    const char *name() const;

                    /**
                     * @brief 返回调度提示，例如：要拆分的维度
                     * @return 一个 @ref IScheduler::Hints 的实例，用于描述调度提示
                     */
                    const BIIScheduler::Hints &scheduler_hint() const;

                private:
                    struct DynamicGemmKernel {
                        const char *name{nullptr};
                        const DataTypeISASelectorPtr is_selected{nullptr};

                        KernelPtr ukernel{nullptr};
                        PackRhsPtr pack_rhs{nullptr};
                        SizeOfPackedRhsPtr size_of_packed_rhs{nullptr};
                        GetWindowPtr get_window{nullptr};
                    };

                    using KernelList = std::vector<DynamicGemmKernel>;
                    using KernelMap = std::map<BIDataType, KernelList>;

                private:
                    /** Chooses a kernel to run and saves it into _kernel data member
                     *
                     * @param[in] selector Selector object based on input and device configuration
                     */
                    void choose_kernel(const BIDataTypeISASelectorData &selector);

                private:
                    const static KernelList fp32_kernels;
                    const static KernelMap kernels;

                    size_t _mws{BIICPPKernel::default_mws};
                    const DynamicGemmKernel *_kernel{nullptr};
                    BIIScheduler::Hints _hint{BIWindow::DimY};
                };
            }
        }
    }
}
