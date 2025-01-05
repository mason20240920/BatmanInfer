//
// Created by Mason on 2025/1/5.
//

#ifndef BATMANINFER_BI_GEMM_ARRAYS_HPP
#define BATMANINFER_BI_GEMM_ARRAYS_HPP

#pragma once

namespace BatmanGemm {
    struct BIIGemmArrays {

        /**
         * @brief 传入指向操作数组的指针及其步幅
         *        这个“通用”版本使用了 void* 指针，但更推荐使用下面由模板类 GemmCommon 提供的版本
         *        因为它使用了具有适当类型的指针。
         *        如果 B 是预先转置的（见下文），那么这里对 B 的设置将被忽略。
         * @param A 指向矩阵 A 的基地址（输入矩阵）
         * @param lda  A 的列步幅（leading dimension of A）
         * @param A_batch_stride 批次间矩阵 A 的步幅, 如果有多个批次，每个批次的 A 在内存中的起始地址间隔
         * @param A_multi_stride 多分量（multi-component）间矩阵 A 的步幅, 用于描述多分量张量中，矩阵 A 在不同分量之间的内存间隔
         * @param B 指向矩阵 B 的基地址（输入矩阵）
         * @param ldb 矩阵 B 的列步幅（leading dimension of B）
         * @param B_multi_stride 多分量间矩阵 B 的步幅
         * @param C 指向矩阵 C 的基地址（结果矩阵）
         * @param ldc 矩阵 C 的列步幅（leading dimension of C）
         * @param C_batch_stride 批次间矩阵 C 的步幅
         * @param C_multi_stride 多分量间矩阵 C 的步幅
         * @param bias 指向偏置向量的基地址
         * @param bias_multi_stride
         */
        virtual void set_arrays_generic(const void *A,
                                        const int lda,
                                        const int A_batch_stride,
                                        const int A_multi_stride,
                                        const void *B,
                                        const int ldb,
                                        const int B_multi_stride, /* batches share B */
                                        void *C,
                                        const int ldc,
                                        const int C_batch_stride,
                                        const int C_multi_stride,
                                        const void *bias,
                                        const int bias_multi_stride) = 0;

        virtual ~BIIGemmArrays() = default;
    };

    template<typename To, typename Tw, typename Tr>
    struct BIGemmArrays : public BIIGemmArrays {
        const To *_Aptr             = nullptr;
        int      _lda               = 0;
        int      _A_batch_stride    = 0;
        int      _A_multi_stride    = 0;
        const Tw *_Bptr             = nullptr;
        int      _ldb               = 0;
        int      _B_multi_stride    = 0;
        Tr       *_Cptr             = nullptr;
        int      _ldc               = 0;
        int      _C_batch_stride    = 0;
        int      _C_multi_stride    = 0;
        const Tr *_bias             = nullptr;
        int      _bias_multi_stride = 0;

        BIGemmArrays() = default;

        BIGemmArrays(const To *A,
                     const int lda,
                     const int A_batch_stride,
                     const int A_multi_stride,
                     const Tw *B,
                     const int ldb,
                     const int B_multi_stride, /* batches share B */
                     Tr *C,
                     const int ldc,
                     const int C_batch_stride,
                     const int C_multi_stride,
                     const Tr *bias,
                     const int bias_multi_stride) /* no row or batch stride needed */
                : _Aptr(A),
                  _lda(lda),
                  _A_batch_stride(A_batch_stride),
                  _A_multi_stride(A_multi_stride),
                  _Bptr(B),
                  _ldb(ldb),
                  _B_multi_stride(B_multi_stride),
                  _Cptr(C),
                  _ldc(ldc),
                  _C_batch_stride(C_batch_stride),
                  _C_multi_stride(C_multi_stride),
                  _bias(bias),
                  _bias_multi_stride(bias_multi_stride) {
        }

        BIGemmArrays(const BIGemmArrays<To, Tw, Tr> &) = default;

        BIGemmArrays &operator=(const BIGemmArrays<To, Tw, Tr> &) = default;

        BIGemmArrays(BIGemmArrays<To, Tw, Tr> &&) = delete;

        BIGemmArrays &operator=(BIGemmArrays<To, Tw, Tr> &&) = delete;

        ~BIGemmArrays() override;

        /**
         * @brief 传入要操作的数组指针及其步长（具有适当类型的模板版本）。
         */
        void set_arrays(const To *A,
                        const int lda,
                        const int A_batch_stride,
                        const int A_multi_stride,
                        const Tw *B,
                        const int ldb,
                        const int B_multi_stride, /* batches share B */
                        Tr *C,
                        const int ldc,
                        const int C_batch_stride,
                        const int C_multi_stride,
                        const Tr *bias,
                        const int bias_multi_stride) /* no row or batch stride needed */
        {
            _Aptr              = A;
            _lda               = lda;
            _A_batch_stride    = A_batch_stride;
            _A_multi_stride    = A_multi_stride;
            _Bptr              = B;
            _ldb               = ldb;
            _B_multi_stride    = B_multi_stride;
            _Cptr              = C;
            _ldc               = ldc;
            _C_batch_stride    = C_batch_stride;
            _C_multi_stride    = C_multi_stride;
            _bias              = bias;
            _bias_multi_stride = bias_multi_stride;
        }

        /**
         * @brief 传入要操作的数组指针及其步长（具有适当类型的模板版本）。
         */
        void set_arrays_generic(const void *A, const int lda, const int A_batch_stride, const int A_multi_stride,
                                const void *B, const int ldb, const int B_multi_stride, void *C, const int ldc,
                                const int C_batch_stride, const int C_multi_stride, const void *bias,
                                const int bias_multi_stride) override {
            set_arrays(static_cast<const To *>(A), lda, A_batch_stride, A_multi_stride, static_cast<const Tw *>(B), ldb,
                       B_multi_stride, static_cast<Tr *>(C), ldc, C_batch_stride, C_multi_stride,
                       static_cast<const Tr *>(bias), bias_multi_stride);
        }
    };
}

#endif //BATMANINFER_BI_GEMM_ARRAYS_HPP
