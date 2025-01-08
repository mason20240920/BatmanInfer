//
// Created by Mason on 2025/1/8.
//

#ifndef BATMANINFER_BI_REGISTERS_HPP
#define BATMANINFER_BI_REGISTERS_HPP

#if defined(ENABLE_FP16_KERNELS)

#if defined(BI_COMPUTE_ENABLE_SVE)
#define REGISTER_FP16_SVE(func_name) &(func_name)
#else /* !defined(BI_COMPUTE_ENABLE_SVE) */
#define REGISTER_FP16_SVE(func_name) nullptr
#endif /* defined(BI_COMPUTE_ENABLE_SVE) */

#if defined(BI_COMPUTE_ENABLE_SVE2)
#define REGISTER_FP16_SVE2(func_name) &(func_name)
#else /* !defined(BI_COMPUTE_ENABLE_SVE2) */
#define REGISTER_FP16_SVE2(func_name) nullptr
#endif /* defined(BI_COMPUTE_ENABLE_SVE2) */

#if defined(BI_COMPUTE_ENABLE_SME2)
#define REGISTER_FP16_SME2(func_name) &(func_name)
#else /* !defined(BI_COMPUTE_ENABLE_SME2) */
#define REGISTER_FP16_SME2(func_name) nullptr
#endif /* defined(BI_COMPUTE_ENABLE_SME2) */

#if defined(BI_COMPUTE_ENABLE_NEON)
#define REGISTER_FP16_NEON(func_name) &(func_name)
#else /* !defined(BI_COMPUTE_ENABLE_NEON) */
#define REGISTER_FP16_NEON(func_name) nullptr
#endif /* defined(BI_COMPUTE_ENABLE_NEON) */

#else /* !defined(ENABLE_FP16_KERNELS) */
#define REGISTER_FP16_NEON(func_name) nullptr
#define REGISTER_FP16_SVE(func_name)  nullptr
#define REGISTER_FP16_SVE2(func_name) nullptr
#define REGISTER_FP16_SME2(func_name) nullptr
#endif /* defined(__BI_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(ENABLE_FP16_KERNELS) */

#if defined(ENABLE_FP32_KERNELS)

#if defined(BI_COMPUTE_ENABLE_SVE)
#define REGISTER_FP32_SVE(func_name) &(func_name)
#else /* !defined(BI_COMPUTE_ENABLE_SVE) */
#define REGISTER_FP32_SVE(func_name) nullptr
#endif /* defined(BI_COMPUTE_ENABLE_SVE) */

#if defined(BI_COMPUTE_ENABLE_SVE2)
#define REGISTER_FP32_SVE2(func_name) &(func_name)
#else /* !defined(BI_COMPUTE_ENABLE_SVE2) */
#define REGISTER_FP32_SVE2(func_name) nullptr
#endif /* defined(BI_COMPUTE_ENABLE_SVE2) */

#if defined(BI_COMPUTE_ENABLE_SME2)
#define REGISTER_FP32_SME2(func_name)           &(func_name)
#define REGISTER_QASYMM8_SME2(func_name)        &(func_name)
#define REGISTER_QASYMM8_SIGNED_SME2(func_name) &(func_name)
#else /* !defined(BI_COMPUTE_ENABLE_SME2) */
#define REGISTER_FP32_SME2(func_name)           nullptr
#define REGISTER_QASYMM8_SME2(func_name)        nullptr
#define REGISTER_QASYMM8_SIGNED_SME2(func_name) nullptr
#endif /* defined(BI_COMPUTE_ENABLE_SME2) */

#if defined(BI_COMPUTE_ENABLE_NEON)
#define REGISTER_FP32_NEON(func_name) &(func_name)
#else /* !defined(BI_COMPUTE_ENABLE_NEON) */
#define REGISTER_FP32_NEON(func_name) nullptr
#endif /* defined(BI_COMPUTE_ENABLE_NEON) */

#else /* defined(ENABLE_FP32_KERNELS) */
#define REGISTER_FP32_NEON(func_name) nullptr
#define REGISTER_FP32_SVE(func_name)  nullptr
#define REGISTER_FP32_SVE2(func_name) nullptr
#define REGISTER_FP32_SME2(func_name) nullptr
#endif /* defined(ENABLE_FP32_KERNELS) */

#if defined(ENABLE_QASYMM8_SIGNED_KERNELS)

#define REGISTER_QASYMM8_SIGNED_NEON(func_name) &(func_name)

#if defined(BI_COMPUTE_ENABLE_SVE)
#define REGISTER_QASYMM8_SIGNED_SVE(func_name) &(func_name)
#else /* !defined(BI_COMPUTE_ENABLE_SVE) */
#define REGISTER_QASYMM8_SIGNED_SVE(func_name) nullptr
#endif /* defined(BI_COMPUTE_ENABLE_SVE) */

#if defined(BI_COMPUTE_ENABLE_SVE2)
#define REGISTER_QASYMM8_SIGNED_SVE2(func_name) &(func_name)
#else /* !defined(BI_COMPUTE_ENABLE_SVE2) */
#define REGISTER_QASYMM8_SIGNED_SVE2(func_name) nullptr
#endif /* defined(BI_COMPUTE_ENABLE_SVE2) */

#if defined(BI_COMPUTE_ENABLE_SME2)
#define REGISTER_QASYMM8_SIGNED_SME2(func_name) &(func_name)
#else /* !defined(BI_COMPUTE_ENABLE_SME2) */
#define REGISTER_QASYMM8_SIGNED_SME2(func_name) nullptr
#endif /* defined(BI_COMPUTE_ENABLE_SME2) */

#else /* defined(ENABLE_QASYMM8_SIGNED_KERNELS) */
#define REGISTER_QASYMM8_SIGNED_NEON(func_name) nullptr
#define REGISTER_QASYMM8_SIGNED_SVE(func_name)  nullptr
#define REGISTER_QASYMM8_SIGNED_SVE2(func_name) nullptr
#define REGISTER_QASYMM8_SIGNED_SME2(func_name) nullptr
#endif /* defined(ENABLE_QASYMM8_SIGNED_KERNELS) */

#if defined(ENABLE_QASYMM8_KERNELS)
#define REGISTER_QASYMM8_NEON(func_name) &(func_name)

#if defined(BI_COMPUTE_ENABLE_SVE)
#define REGISTER_QASYMM8_SVE(func_name) &(func_name)
#else /* !defined(BI_COMPUTE_ENABLE_SVE) */
#define REGISTER_QASYMM8_SVE(func_name) nullptr
#endif /* defined(BI_COMPUTE_ENABLE_SVE) */

#if defined(BI_COMPUTE_ENABLE_SVE2)
#define REGISTER_QASYMM8_SVE2(func_name) &(func_name)
#else /* !defined(BI_COMPUTE_ENABLE_SVE2) */
#define REGISTER_QASYMM8_SVE2(func_name) nullptr
#endif /* defined(BI_COMPUTE_ENABLE_SVE2) */

#if defined(BI_COMPUTE_ENABLE_SME2)
#define REGISTER_QASYMM8_SME2(func_name) &(func_name)
#else /* !defined(BI_COMPUTE_ENABLE_SME2) */
#define REGISTER_QASYMM8_SME2(func_name) nullptr
#endif /* defined(BI_COMPUTE_ENABLE_SME2) */

#else /* defined(ENABLE_QASYMM8_KERNELS) */
#define REGISTER_QASYMM8_NEON(func_name) nullptr
#define REGISTER_QASYMM8_SVE(func_name)  nullptr
#define REGISTER_QASYMM8_SVE2(func_name) nullptr
#define REGISTER_QASYMM8_SME2(func_name) nullptr
#endif /* defined(ENABLE_QASYMM8_KERNELS) */

#if defined(ENABLE_QSYMM16_KERNELS)

#define REGISTER_QSYMM16_NEON(func_name) &(func_name)

#if defined(BI_COMPUTE_ENABLE_SVE)
#define REGISTER_QSYMM16_SVE(func_name) &(func_name)
#else /* !defined(BI_COMPUTE_ENABLE_SVE) */
#define REGISTER_QSYMM16_SVE(func_name) nullptr
#endif /* defined(BI_COMPUTE_ENABLE_SVE) */

#if defined(BI_COMPUTE_ENABLE_SVE2)
#define REGISTER_QSYMM16_SVE2(func_name) &(func_name)
#else /* !defined(BI_COMPUTE_ENABLE_SVE2) */
#define REGISTER_QSYMM16_SVE2(func_name) nullptr
#endif /* defined(BI_COMPUTE_ENABLE_SVE2) */

#else /* defined(ENABLE_QSYMM16_KERNELS) */
#define REGISTER_QSYMM16_NEON(func_name) nullptr
#define REGISTER_QSYMM16_SVE(func_name)  nullptr
#define REGISTER_QSYMM16_SVE2(func_name) nullptr
#endif /* defined(ENABLE_QSYMM16_KERNELS) */

#if defined(ENABLE_QASYMM8_KERNELS) || defined(ENABLE_QASYMM8_SIGNED_KERNELS)
#define REGISTER_Q8_NEON(func_name) &(func_name)
#else /* !defined(ENABLE_QASYMM8_KERNELS) && !defined(ENABLE_QASYMM8_SIGNED_KERNELS) */
#define REGISTER_Q8_NEON(func_name) nullptr
#endif /* defined(ENABLE_QASYMM8_KERNELS) || defined(ENABLE_QASYMM8_SIGNED_KERNELS) */

#if defined(ENABLE_INTEGER_KERNELS)

#if defined(BI_COMPUTE_ENABLE_SVE)
#define REGISTER_INTEGER_SVE(func_name) &(func_name)
#else /* !defined(BI_COMPUTE_ENABLE_SVE) */
#define REGISTER_INTEGER_SVE(func_name) nullptr
#endif /* defined(BI_COMPUTE_ENABLE_SVE) */

#if defined(BI_COMPUTE_ENABLE_SVE2)
#define REGISTER_INTEGER_SVE2(func_name) &(func_name)
#else /* !defined(BI_COMPUTE_ENABLE_SVE2) */
#define REGISTER_INTEGER_SVE2(func_name) nullptr
#endif /* defined(BI_COMPUTE_ENABLE_SVE2) */

#if defined(BI_COMPUTE_ENABLE_NEON)
#define REGISTER_INTEGER_NEON(func_name) &(func_name)
#else /* !defined(BI_COMPUTE_ENABLE_NEON) */
#define REGISTER_INTEGER_NEON(func_name) nullptr
#endif /* defined(BI_COMPUTE_ENABLE_NEON) */

#else /* defined(ENABLE_INTEGER_KERNELS) */
#define REGISTER_INTEGER_NEON(func_name) nullptr
#define REGISTER_INTEGER_SVE(func_name)  nullptr
#define REGISTER_INTEGER_SVE2(func_name) nullptr
#endif /* defined(ENABLE_INTEGER_KERNELS) */

#if defined(BI_COMPUTE_ENABLE_BF16)
#define REGISTER_BF16_NEON(func_name) &(func_name)
#if defined(BI_COMPUTE_ENABLE_SVE)
#define REGISTER_BF16_SVE(func_name) &(func_name)
#endif /* !defined(BI_COMPUTE_ENABLE_SVE)*/
#else  /* !(defined(BI_COMPUTE_ENABLE_BF16))*/
#define REGISTER_BF16_NEON(func_name) nullptr
#if defined(BI_COMPUTE_ENABLE_SVE)
#define REGISTER_BF16_SVE(func_name) nullptr
#endif /* !defined(BI_COMPUTE_ENABLE_SVE)*/
#endif /* defined(BI_COMPUTE_ENABLE_BF16)*/

#endif //BATMANINFER_BI_REGISTERS_HPP
