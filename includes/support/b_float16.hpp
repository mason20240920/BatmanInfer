//
// Created by Mason on 2025/1/2.
//

#ifndef BATMANINFER_B_FLOAT16_HPP
#define BATMANINFER_B_FLOAT16_HPP

#include <cstdint>
#include <cstring>
#include <ostream>

namespace BatmanInfer {
    namespace {
        /**
         * @brief  Convert float to bfloat16 in a portable way that works on older hardware
         * @param v Floating-point value to convert to bfloat
         * @return Converted value
         */
        inline uint16_t portable_float_to_bf16(const float v) {
            // 将输入的 float 类型的值 `v` 的地址 reinterpret_cast 转换为指向 uint32_t 类型的指针。
            // 这样可以直接以 uint32_t 的形式访问 float 的二进制表示。
            const uint32_t *from_ptr = reinterpret_cast<const uint32_t *>(&v);

            // 将 32 位浮点数的高 16 位提取出来，存储到 `res` 中。
            // BF16（BFloat16）是 16 位浮点数格式，其中高 16 位是原始 float 的高 16 位。
            uint16_t res = (*from_ptr >> 16);

            // 提取原始 float 的低 16 位（用于舍入判断）。
            const uint16_t error = (*from_ptr & 0x0000ffff);

            // 提取 BF16 的最低有效位（LSB），用于处理舍入时的“向偶数舍入”规则。
            uint16_t bf_l = res & 0x0001;

            // 执行舍入操作：
            // 1. 如果低 16 位（error）大于 0x8000，表示需要向上舍入。
            // 2. 如果低 16 位等于 0x8000，并且 BF16 的最低有效位（bf_l）为 1，
            //    按照“向偶数舍入”（round to nearest even）的规则，也需要向上舍入。
            if ((error > 0x8000) || ((error == 0x8000) && (bf_l != 0))) {
                // 如果需要舍入，则将高 16 位（res）加 1。
                res += 1;
            }

            // 返回 BF16 格式的结果（高 16 位，经过舍入）。
            return res;
        }


        /**
         * @brief  Convert float to bfloat16
         * @param v Floating-point value to convert to bfloat
         * @return Converted value
         */
        inline uint16_t float_to_bf16(const float v) {
#if defined(BI_COMPUTE_ENABLE_BF16)
            const uint32_t *fromptr = reinterpret_cast<const uint32_t *>(&v);
            uint16_t res;

            // 从 fromptr 指向的地址（即 v 的地址）加载 32 位浮点数到 SIMD 寄存器 s0
            // BFCVT 指令：将 32 位浮点数（s0）转换为 16 位 BF16 浮点数（h0）
            // h0 是一个 16 位浮点寄存器，属于 NEON 的寄存器组
            // 转换过程中会执行截断或舍入操作（具体规则由硬件定义，通常是“向偶数舍入”规则）
            // 将转换后的 16 位 BF16 浮点数（h0）存储到 toptr 指向的地址（即 &res）
            // [fromptr] "r"(fromptr): 将 fromptr 的值（v 的地址）作为寄存器输入，传递给汇编代码
            // [toptr] "r"(&res): 将 res 的地址作为寄存器输入，传递给汇编代码
            // "v0": 表示汇编代码会使用 SIMD 浮点寄存器组中的 v0
            // "memory": 表示汇编代码会修改内存，编译器需要保证内存的同步
            __asm __volatile("ldr    s0, [%[fromptr]]\n"
                             ".inst    0x1e634000\n" // BFCVT h0, s0
                             "str    h0, [%[toptr]]\n"
                             :
                             : [fromptr] "r"(fromptr), [toptr] "r"(&res)
                             : "v0", "memory");
            return res;
#else /* defined(ARM_COMPUTE_ENABLE_BF16) */
            return portable_float_to_bf16(v);
#endif
        }

        /**
         * @brief Convert bfloat16 to float
         * @param v v Bfloat16 value to convert to float
         * @return Converted value
         */
        inline float bf16_to_float(const uint16_t &v)
        {
            const uint32_t lv = (v << 16);
            float fp;
            memcpy(&fp, &lv, sizeof(lv));
            return fp;
        }
    }

    /**
     * @brief Brain floating point representation class
     */
    class bfloat16 final {
    public:
        /**
         * @brief Default constructor
         */
        bfloat16() : value(0)
        {

        }

        /**
         * @brief Constructor
         * @param v Floating-point value
         */
        explicit bfloat16(float v): value(float_to_bf16(v))
        {

        }

        /**
         * @brief Constructor
         * @param v  Floating-point value
         * @param portable bool to indicate the conversion is to be done in a backward compatible way
         */
        bfloat16(float v, float portable) : value(0)
        {
            value = portable ? portable_float_to_bf16(v) : float_to_bf16(v);
        }

        /**
         * @brief  Assignment operator
         * @param v Floating point value to assign
         * @return The updated object
         */
        bfloat16 &operator=(float v)
        {
            value = float_to_bf16(v);
            return *this;
        }

        /**
         * @brief Floating point conversion operator
         * @return Floating point representation of the value
         */
        operator float() const
        {
            return bf16_to_float(value);
        }

        /**
         * @brief Lowest representative value
         * @return  Returns the lowest finite value representable by bfloat16
         */
        static bfloat16 lowest()
        {
            bfloat16 val;
            val.value = 0xFF7F;
            return val;
        }

        /**
         * @brief Largest representative value
         * @return  Returns the largest finite value representable by bfloat16
         */
        static bfloat16 max()
        {
            bfloat16 val;
            val.value = 0x7F7F;
            return val;
        }

        bfloat16 &operator+=(float v)
        {
            value = float_to_bf16(bf16_to_float(value) + v);
            return *this;
        }

        friend std::ostream &operator<<(std::ostream &os, const bfloat16 &arg);

    private:
        uint16_t value;
    };
}

#endif //BATMANINFER_B_FLOAT16_HPP
