//
// Created by Mason on 2025/1/14.
//

#pragma once

#ifdef __aarch64__
// 针对 AArch64（64 位 ARM 架构）的预取宏定义
// 使用 PRFM 指令进行预取，优化数据访问性能

// 预取到 L1 缓存，保持缓存行
#define ASM_PREFETCH(address)    "PRFM PLDL1KEEP, " address "\n"
// 未对齐地址的预取到 L1 缓存
#define ASM_PREFETCHU(address)   "PRFUM PLDL1KEEP, " address "\n"
// 预取到 L2 缓存，保持缓存行
#define ASM_PREFETCHL2(address)  "PRFM PLDL2KEEP, " address "\n"
// 写入预取到 L1 缓存
#define ASM_PREFETCHW(address)   "PRFM PSTL1KEEP, " address "\n"
// 写入预取到 L2 缓存
#define ASM_PREFETCHWL2(address) "PRFM PSTL2KEEP, " address "\n"

// 如果不需要预取，可以将以下宏定义注释取消（禁用预取）
// #define ASM_PREFETCH(address) ""

#else
// 针对 AArch32（32 位 ARM 架构）的预取宏定义
// 使用 PLD 和 PLDW 指令进行预取

// 预取数据
#define ASM_PREFETCH(address)     "PLD " address "\n"
// 写入预取
#define ASM_PREFETCHW(address)    "PLDW " address "\n"

#endif

/*
 * Do some prefetches.
 */

/*
 * 以下为预取函数模板的实现。
 * 提供了从 1 次到 6 次预取的函数，适应不同的数据访问需求。
 * 预取的目的是减少缓存未命中（cache miss）的开销，从而提升性能。
 */
template<typename T>
static inline void prefetch_6x(const T *pfp) {
    // 使用内联汇编进行 6 次连续预取，每次间隔 64 字节
    __asm __volatile (
            ASM_PREFETCH("[%[pfp]]")          // 预取 pfp 指针指向的地址
            ASM_PREFETCH("[%[pfp], #64]")    // 预取 pfp + 64 字节
            ASM_PREFETCH("[%[pfp], #128]")   // 预取 pfp + 128 字节
            ASM_PREFETCH("[%[pfp], #192]")   // 预取 pfp + 192 字节
            ASM_PREFETCH("[%[pfp], #256]")   // 预取 pfp + 256 字节
            ASM_PREFETCH("[%[pfp], #320]")   // 预取 pfp + 320 字节
            :
            : [pfp] "r"(pfp)      // 将 pfp 作为输入寄存器
    : "memory"                    // 声明该操作可能影响内存
    );
}

template<typename T>
static inline void prefetch_5x(const T *pfp) {
    // 使用内联汇编进行 5 次连续预取，每次间隔 64 字节
    __asm __volatile (
            ASM_PREFETCH("[%[pfp]]")
            ASM_PREFETCH("[%[pfp], #64]")
            ASM_PREFETCH("[%[pfp], #128]")
            ASM_PREFETCH("[%[pfp], #192]")
            ASM_PREFETCH("[%[pfp], #256]")
            :
            : [pfp] "r"(pfp)
    : "memory"
    );
}

template<typename T>
static inline void prefetch_4x(const T *pfp) {
    // 使用内联汇编进行 4 次连续预取，每次间隔 64 字节
    __asm __volatile (
            ASM_PREFETCH("[%[pfp]]")
            ASM_PREFETCH("[%[pfp], #64]")
            ASM_PREFETCH("[%[pfp], #128]")
            ASM_PREFETCH("[%[pfp], #192]")
            :
            : [pfp] "r"(pfp)
    : "memory"
    );
}

template<typename T>
static inline void prefetch_3x(const T *pfp) {
    // 使用内联汇编进行 3 次连续预取，每次间隔 64 字节
    __asm __volatile (
            ASM_PREFETCH("[%[pfp]]")
            ASM_PREFETCH("[%[pfp], #64]")
            ASM_PREFETCH("[%[pfp], #128]")
            :
            : [pfp] "r"(pfp)
    : "memory"
    );
}

template<typename T>
static inline void prefetch_2x(const T *pfp) {
    // 使用内联汇编进行 2 次连续预取，每次间隔 64 字节
    __asm __volatile (
            ASM_PREFETCH("[%[pfp]]")
            ASM_PREFETCH("[%[pfp], #64]")
            :
            : [pfp] "r"(pfp)
    : "memory"
    );
}

template<typename T>
static inline void prefetch_1x(const T *pfp) {
    // 使用内联汇编进行 1 次预取
    __asm __volatile (
            ASM_PREFETCH("[%[pfp]]")
            :
            : [pfp] "r"(pfp)
    : "memory"
    );
}
