//
// Created by Mason on 2025/1/14.
//

#pragma once

#ifdef __arm__

#include <neon/neon_defines.h>

#include "../asmlib.hpp"

template<>
void interleave_block<6, 1, VLType::None, false>(
    float * &outptr, const float * const * in, size_t width, size_t height,
    size_t row_offset, bool
)
{
    // 获取输入矩阵的每一行指针，并根据行偏移量调整起始地址
    const float *inptr0 = in[0] + row_offset;
    const float *inptr1 = in[1] + row_offset;
    const float *inptr2 = in[2] + row_offset;
    const float *inptr3 = in[3] + row_offset;
    const float *inptr4 = in[4] + row_offset;
    const float *inptr5 = in[5] + row_offset;

    // 处理矩阵行数不足 6 行的情况
    // 如果 height 小于 6，则将缺失的行指针指向第一行（inptr0）
    // 这样可以避免访问非法内存地址
    switch (height) {
        case 1:
            inptr1 = inptr0;  // 如果只有 1 行，所有行都指向 inptr0
            // fall through
        case 2:
            inptr2 = inptr0;  // 如果只有 2 行，剩余行都指向 inptr0
            // fall through
        case 3:
            inptr3 = inptr0;  // 如果只有 3 行，剩余行都指向 inptr0
            // fall through
        case 4:
            inptr4 = inptr0;  // 如果只有 4 行，剩余行都指向 inptr0
            // fall through
        case 5:
            inptr5 = inptr0;  // 如果只有 5 行，剩余行都指向 inptr0
            // fall through
        default:
        case 6:
            break;  // 如果有 6 行，正常处理
    }

    // 如果需要，可以启用预取操作以优化内存访问性能
    // 这里注释掉了预取代码
    // prefetch_2x(inptr0);
    // prefetch_2x(inptr1);
    // prefetch_2x(inptr2);
    // prefetch_2x(inptr3);
    // prefetch_2x(inptr4);
    // prefetch_2x(inptr5);

    // 主循环：每次处理 8 列数据
    for (;width>7;width-=8) {
        __asm __volatile (
            // 从每一行加载 8 个元素（2 个向量）
            "VLD1.32    {d0-d3}, [%[inptr0]]!\n"   // 从 inptr0 加载 8 个元素到 q0 和 q1
            "VLD1.32    {d4-d7}, [%[inptr1]]!\n"   // 从 inptr1 加载 8 个元素到 q2 和 q3
            "VLD1.32    {d8-d11}, [%[inptr2]]!\n"  // 从 inptr2 加载 8 个元素到 q4 和 q5
            "VZIP.32    q0, q4\n"     // 交错 q0 和 q4：q0=A0C0A1C1, q4=A2C2A3C3
            "VLD1.32    {d12-d15}, [%[inptr3]]!\n" // 从 inptr3 加载 8 个元素到 q6 和 q7
            "VZIP.32    q2, q6\n"     // 交错 q2 和 q6：q2=B0D0B1D1, q6=B2D2B3D3
            "VLD1.32    {d16-d19}, [%[inptr4]]!\n" // 从 inptr4 加载 8 个元素到 q8 和 q9
            "VLD1.32    {d20-d23}, [%[inptr5]]!\n" // 从 inptr5 加载 8 个元素到 q10 和 q11
            "VZIP.32    q8, q10\n"    // 交错 q8 和 q10：q8=E0F0E1F1, q10=E2F2E3F3
            ASM_PREFETCH("[%[inptr0], #128]")      // 预取 inptr0 的下一部分数据
            "VZIP.32    q0, q2\n"     // 交错 q0 和 q2：q0=A0B0C0D0, q2=A1B1C1D1

            // 第一批存储
            "VST1.32    {d0-d1}, [%[outptr]]!\n"   // 将 q0 的前半部分存储到 outptr
            "VST1.32    {d16}, [%[outptr]]!\n"    // 将 q8 的前半部分存储到 outptr

            "VZIP.32    q4, q6\n"     // 交错 q4 和 q6：q4=A2B2C2D2, q6=A3B3C3D3

            // 第二批存储
            "VST1.32    {d4-d5}, [%[outptr]]!\n"   // 将 q4 的前半部分存储到 outptr
            "VZIP.32    q1, q5\n"                 // 交错 q1 和 q5
            ASM_PREFETCH("[%[inptr1], #128]")      // 预取 inptr1 的下一部分数据
            "VST1.32    {d17}, [%[outptr]]!\n"    // 将 q8 的后半部分存储到 outptr
            "VZIP.32    q3, q7\n"                 // 交错 q3 和 q7

            // 第三批存储
            "VZIP.32    q9, q11\n"                // 交错 q9 和 q11
            "VST1.32    {d8-d9}, [%[outptr]]!\n"  // 将 q9 的前半部分存储到 outptr
            "VZIP.32    q1, q3\n"                 // 交错 q1 和 q3
            ASM_PREFETCH("[%[inptr2], #128]")      // 预取 inptr2 的下一部分数据
            "VST1.32    {d20}, [%[outptr]]!\n"    // 将 q10 的前半部分存储到 outptr

            // 第四批存储
            "VZIP.32    q5, q7\n"                 // 交错 q5 和 q7
            "VST1.32    {d12-d13}, [%[outptr]]!\n"// 将 q6 的前半部分存储到 outptr
            ASM_PREFETCH("[%[inptr3], #128]")      // 预取 inptr3 的下一部分数据
            "VST1.32    {d21}, [%[outptr]]!\n"    // 将 q10 的后半部分存储到 outptr

            // 第五批存储
            "VST1.32    {d2-d3}, [%[outptr]]!\n"  // 将 q1 的后半部分存储到 outptr
            ASM_PREFETCH("[%[inptr4], #128]")      // 预取 inptr4 的下一部分数据
            "VST1.32    {d18}, [%[outptr]]!\n"    // 将 q9 的后半部分存储到 outptr

            // 第六批存储
            "VST1.32    {d6-d7}, [%[outptr]]!\n"  // 将 q3 的后半部分存储到 outptr
            ASM_PREFETCH("[%[inptr5], #128]")      // 预取 inptr5 的下一部分数据
            "VST1.32    {d19}, [%[outptr]]!\n"    // 将 q11 的后半部分存储到 outptr

            // 第七批存储
            "VST1.32    {d10-d11}, [%[outptr]]!\n"// 将 q5 的后半部分存储到 outptr
            "VST1.32    {d22}, [%[outptr]]!\n"    // 将 q11 的后半部分存储到 outptr

            // 第八批存储
            "VST1.32    {d14-d15}, [%[outptr]]!\n"// 将 q7 的后半部分存储到 outptr
            "VST1.32    {d23}, [%[outptr]]!\n"    // 将 q11 的后半部分存储到 outptr

            : [inptr0] "+r" (inptr0), [inptr1] "+r" (inptr1), [inptr2] "+r" (inptr2), [inptr3] "+r" (inptr3),
              [inptr4] "+r" (inptr4), [inptr5] "+r" (inptr5), [outptr] "+r" (outptr)
            :
            : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "memory"
        );
    }

    // 剩余列的处理：逐列交错存储
    for (;width>0;width--) {
        *outptr++ = *inptr0++;  // 存储 inptr0 的当前元素
        *outptr++ = *inptr1++;  // 存储 inptr1 的当前元素
        *outptr++ = *inptr2++;  // 存储 inptr2 的当前元素
        *outptr++ = *inptr3++;  // 存储 inptr3 的当前元素
        *outptr++ = *inptr4++;  // 存储 inptr4 的当前元素
        *outptr++ = *inptr5++;  // 存储 inptr5 的当前元素
    }
}

#endif  // __arm__