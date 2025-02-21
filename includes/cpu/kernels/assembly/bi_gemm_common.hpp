//
// Created by Mason on 2025/1/5.
//

#ifndef BATMANINFER_BI_GEMM_COMMON_HPP
#define BATMANINFER_BI_GEMM_COMMON_HPP

#pragma once

#include "bi_convolution_parameters.hpp"
#include "bi_gemm_arrays.hpp"
#include "bi_nd_range.hpp"
#include <cstddef>

namespace BatmanGemm {
    // Avoid circular dependency with arm_gemm.hpp
    struct GemmConfig;
    struct Requantize32;

    /**
     * @brief 用于 GEMM/GEMV 函数的抽象类
     *        GEMV: General Matrix-Vector Multiplication (通用矩阵-向量乘法)
     *        GEMM 的实现可能是以下几种方式：
     *        1. "原生" (native)：不需要任何输入数据的重排（permutation）
     *        2. "预转置" (pre_transposed)：需要在计算前对输入数据进行重排。
     *        > 解释: 比如矩阵A是列优先, 那么可以修改硬件要求的行优先的排列方式
     *        3. 需要工作空间 (working space)：在计算过程中动态进行数据重排。
     *        该接口应支持所有这些实现方式。
     *
     *        | 实现方式 | 数据预处理 | 计算效率
     *
     *        实际的 BIGemmCommon 类是一个基于操作数类型和返回值类型的模板类。
     *        而这个类是一个独立于这些类型的接口类。
     */
    class BIIGemmCommon {
    public:

        /**
         * @brief 设置操作数组及其步长的指针。
         *        这个“通用”版本使用的是 void * 指针，推荐使用下面由模板化的 BIGemmCommon 提供的版本，
         *        因为它可以接受适当类型的指针。
         *        如果 B 是预转置的（见下文），那么这里对 B 的设置将被忽略
         * @param A
         * @param lda
         * @param A_batch_stride
         * @param A_multi_stride
         * @param B
         * @param ldb
         * @param B_multi_stride
         * @param C
         * @param ldc
         * @param C_batch_stride
         * @param C_multi_stride
         * @param bias
         * @param bias_multi_stride
         */
        virtual void set_arrays_generic(const void *A,
                                        const int lda,
                                        const int A_batch_stride,
                                        const int A_multi_stride,
                                        const void *B,
                                        const int ldb,
                /* batches share B */   const int B_multi_stride,
                                        void *C,
                                        const int ldc,
                                        const int C_batch_stride,
                                        const int C_multi_stride,
                                        const void *bias,
                /* no row or batch stride needed */ const int bias_multi_stride) = 0;

        /**
         * @brief
         * @return 一个 ndrange，它包含可以被分解并并行化的计算空间范围
         */
        virtual ndrange_t get_window_size() const = 0;

        /**
         * @brief 最大线程数在创建 GEMM 时指定。
         *        一些实现需要知道实际运行的线程数才能正常工作。
         *
         *        在某些情况下，在创建 GEMM 后需要减少线程数（例如，没有足够的工作量可以分配到线程中）。
         *        这个方法允许设置实际运行的线程数（必须小于或等于最大线程数）。
         *
         *        这个方法有一个空的默认实现，因为不关心线程数的 GEMM 可以安全地忽略它。
         */
        virtual void set_nthreads(int) {};

        /**
         * @brief 判断当前 GEMM 是否支持动态调度
         * @return
         */
        virtual bool supports_dynamic_scheduling() const {
            return false;
        }


        /** 主执行函数
         * @param [in] work_range     指定需要计算的工作范围，总范围由 get_window_size() 定义
         * @param [in] thread_locator 指定线程空间中的位置
         * @param [in] threadid       唯一的线程 ID
         */
        virtual void execute(const ndcoord_t &work_range, const ndcoord_t &thread_locator, int threadid) = 0;

        /*** 工作空间接口（可选） ***/
        /* 所需的临时工作空间的总字节数。如果为零，则无需调用 set_working_space()。*/
        virtual size_t get_working_size() const {
            return 0;
        }

        /* 提供工作空间缓冲区——传入的 void * 必须在任何 execute 调用期间保持分配状态。*/
        virtual void set_working_space(void *) {};

        /*** "预转置" 接口（可选） ***/
        /* 判断当前对象是否设置为预转置模式。如果是，则在调用 execute() 之前需要调用 pretranspose_array()。*/
        virtual bool B_is_pretransposed() const {
            return false;
        }

        /* 判断是否仍然需要进行预转置 */
        virtual bool B_pretranspose_required() const {
            return false;
        }

        /* 判断预转置是否接受转置标志 */
        virtual bool B_pretranspose_supports_transpose() const {
            return false;
        }

        /* 获取预转置数组所需的总字节数 */
        virtual size_t get_B_pretransposed_array_size() const {
            return 0;
        }

        /* 获取线程分块的工作量 */
        virtual size_t get_B_pretranspose_window_size() const {
            return 1;
        }
        /* 执行预转置——参数包括输出、输入、输入行步幅和输入多步幅。*/
        /* 该方法的“真实”版本依赖于模板化的操作数类型（见下文）。*/
        virtual void pretranspose_B_array_generic(void *, const void *, const int, const int, bool) = 0;

        /* 带有窗口起始/结束参数的多线程版本 */
        virtual void pretranspose_B_array_part_generic(void *, const void *, const int, const int, bool, const size_t,
                                                       const size_t) = 0;

        /* 设置预转置数据——传入的 void * 必须是之前通过 pretranspose_B_array() 为相同或类似的 GEMM 设置的。*/
        virtual void set_pretransposed_B_data(void *) {
        }

        /*** "量化偏置" 接口（可选） ***/
        /* 为量化 GEMM 设置偏置向量 */
        virtual void set_quantized_bias(const int32_t *, size_t) {
        }

        /*** 间接参数接口（可选） ***/
        /* 设置间接表。间接表包含每个内核点的若干值，以及一个密集打包的指针数组，
         * 表示 multis * batches * kernel_points。
         */
        virtual void set_indirect_parameters_generic(size_t, const void *const *const *) {
        }

        /*** "量化更新" 接口（可选） ***/
        /* 在运行时更新量化参数 */
        virtual void update_quantization_parameters(const Requantize32 &) {
        }

        /*** 卷积接口（可选） ***/
        /* 设置卷积参数 */
        virtual void set_convolution_parameters(BIConvolutionParameters) {
        }

        /*** 反量化缩放接口（可选） ***/
        /* 设置从 int 转换到 float 的反量化缩放参数（float out = scale * float(int out)） */
        virtual void set_dequantize_scale(const float) {
        }

        /*** 自省接口 ***/
        /* 获取当前 GEMM 的配置 */
        virtual GemmConfig get_config() = 0;

        // 析构函数
        virtual ~BIIGemmCommon() {
        }
    };

    /* "真实"的 BIGemmCommon 类，它是基于操作数和返回值类型的模板化类。
 *
 * 除了提供用于操作操作数和返回值数据的正确类型版本的函数外，
 * 这个类还提供了一个默认的 `set_arrays` 实现，用于将提供的参数捕获到受保护的类成员中，
 * 因为几乎所有的实现都需要这些参数。
 */
    template<typename To, typename Tw, typename Tr>
    class BIGemmCommon : public BIIGemmCommon {
    protected:
        BIGemmArrays<To, Tw, Tr> _gemm_array{};

    public:
        /* 设置用于操作的数组指针及其步长（模板化版本，使用适当的类型）。 */
        void set_gemm_arrays(BIGemmArrays<To, Tw, Tr> &ga) {
            _gemm_array = ga;
        }

        const BIGemmArrays<To, Tw, Tr> &get_gemm_arrays() const {
            return _gemm_array;
        }

        /**
         * 动态设置矩阵M的大小
         * @param M_size
         */
        virtual void set_dynamic_M_size(int M_size) = 0;

        /* 设置数组的函数实现，用于接收正确类型的参数。 */
        virtual void set_arrays(const To *A,
                                const int lda,
                                const int A_batch_stride,
                                const int A_multi_stride,
                                const Tw *B,
                                const int ldb,
                /* batches share B */ const int B_multi_stride,
                                Tr *C,
                                const int ldc,
                                const int C_batch_stride,
                                const int C_multi_stride,
                                const Tr *bias,
                /* no row or batch stride needed */ const int bias_multi_stride) {
            _gemm_array.set_arrays(A, lda, A_batch_stride, A_multi_stride, B, ldb, B_multi_stride, C, ldc,
                                   C_batch_stride,
                                   C_multi_stride, bias, bias_multi_stride);
        }

        /* 使用 void * 参数的重载实现，将其参数转换为适当的类型。 */
        void set_arrays_generic(const void *A,
                                const int lda,
                                const int A_batch_stride,
                                const int A_multi_stride,
                                const void *B,
                                const int ldb,
                /* batches share B */ const int B_multi_stride,
                                void *C,
                                const int ldc,
                                const int C_batch_stride,
                                const int C_multi_stride,
                                const void *bias,
                /* no row or batch stride needed */ const int bias_multi_stride) override {
            set_arrays(static_cast<const To *>(A), lda, A_batch_stride, A_multi_stride, static_cast<const Tw *>(B), ldb,
                       B_multi_stride, static_cast<Tr *>(C), ldc, C_batch_stride, C_multi_stride,
                       static_cast<const Tr *>(bias), bias_multi_stride);
        }

        /*** "预转置" 接口 ***/

        /* 计算所有列的列和 */
        virtual void requantize_bias(void *, const Tw *, const int, const int) {};

        /* 执行预转置 - 传入的 void * 指针在任何 execute 调用期间必须保持分配状态。 */
        /* 参数为：输出缓冲区指针、源指针、源行步幅、源多步幅。 */
        virtual void pretranspose_B_array(void *, const Tw *, const int, const int, bool) {};

        /* 使用 void * 参数的重载实现，将其参数转换为适当的类型。 */
        void pretranspose_B_array_generic(
                void *out, const void *in, const int row_stride, const int multi_stride, bool transposed) override {
            pretranspose_B_array(out, static_cast<const Tw *>(in), row_stride, multi_stride, transposed);
        }

        /* 上述函数的多线程版本。
         * 线程接口的回退/向后兼容版本将窗口大小设置为 1，并调用非线程化的函数来完成工作。
         * 这是合法的，因为当窗口大小为 1 时，start 和 end 的唯一合法值分别为 0 和 1。
         */
        virtual void pretranspose_B_array_part(
                void *out, const Tw *in, const int row_stride, const int multi_stride, bool transposed, size_t,
                size_t) {
            pretranspose_B_array(out, in, row_stride, multi_stride, transposed);
        };

        /**
         * @brief
         * @param out 输出缓冲区指针
         * @param in  输入矩阵指针
         * @param row_stride  行步幅（矩阵行之间的存储间隔）
         * @param multi_stride 多矩阵步幅（批次矩阵之间的存储间隔）
         * @param transposed 是否需要转置
         * @param start 当前线程负责的起始工作范围
         * @param end 当前线程负责的结束工作范围
         */
        void pretranspose_B_array_part_generic(void *out,
                                               const void *in,
                                               const int row_stride,
                                               const int multi_stride,
                                               bool transposed,
                                               size_t start,
                                               size_t end) override {
            pretranspose_B_array_part(out, static_cast<const Tw *>(in), row_stride, multi_stride, transposed, start,
                                      end);
        }

        /*** 间接参数接口 ***/
        virtual void set_indirect_parameters(size_t, const To *const *const *) {
        }

        void set_indirect_parameters_generic(size_t sz, const void *const *const *ptr) override {
            set_indirect_parameters(sz, reinterpret_cast<const To *const *const *>(ptr));
        }

        /** 无状态版本的主执行函数
         * @param [in] work_range     指定需要计算的工作范围，总范围由 get_window_size() 定义
         * @param [in] thread_locator 指定线程空间中的位置
         * @param [in] threadid       唯一的线程 ID
         * @param [out] GemmArrays    包含输入/输出地址和步长信息的结构体
         */
        virtual void execute_stateless(const ndcoord_t &work_range,
                                       const ndcoord_t &thread_locator,
                                       int threadid,
                                       BIGemmArrays<To, Tw, Tr> &gemm_array) = 0;
    };

}

#endif //BATMANINFER_BI_GEMM_COMMON_HPP
