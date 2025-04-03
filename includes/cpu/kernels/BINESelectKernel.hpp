//
// Created by Mason on 2025/4/3.
//

#pragma once

#include <data/core/bi_types.hpp>
#include <data/core/neon/bi_i_ne_kernel.hpp>

namespace BatmanInfer {
    class BIITensor;

    /** Interface for the select kernel
 *
 * Select is computed by:
 * @f[ output(i) = condition(i) ? x(i) : y(i) @f]
 *
 */
    class BINESelectKernel : public BIINEKernel {
    public:
        const char *name() const override {
            return "BINESelectKernel";
        }

        /** Default constructor */
        BINESelectKernel();

        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BINESelectKernel(const BINESelectKernel &) = delete;

        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BINESelectKernel &operator=(const BINESelectKernel &) = delete;

        /** Allow instances of this class to be moved */
        BINESelectKernel(BINESelectKernel &&) = default;

        /** Allow instances of this class to be moved */
        BINESelectKernel &operator=(BINESelectKernel &&) = default;

        /** Default destructor */
        ~BINESelectKernel() = default;

        /** Common signature for all the specialised elementwise functions
         *
         * @param[in]  c      Condition input tensor. Data types supported: U8.
         * @param[in]  x      First input tensor. Data types supported: All.
         * @param[out] y      Second input tensor. Data types supported: Same as @p x
         * @param[in]  output Output tensor. Data types supported: Same as @p x
         */
        void configure(const BIITensor *c, const BIITensor *x, const BIITensor *y, BIITensor *output);

        /** Validate the argument passed to the kernel
         *
         * @param[in] c      Condition input tensor. Data types supported: U8.
         * @param[in] x      First input tensor. Data types supported: All.
         * @param[in] y      Second input tensor. Data types supported: Same as @p x
         * @param[in] output Output tensor. Data types supported: Same as @p x.
         *
         * @return a status
         */
        static BIStatus validate(const BIITensorInfo *c, const BIITensorInfo *x, const BIITensorInfo *y,
                                 const BIITensorInfo *output);

        // Inherited methods overridden:
        void run(const BIWindow &window, const ThreadInfo &info) override;

    private:
        const BIITensor *_c; /**< Condition tensor */
        const BIITensor *_x; /**< Source tensor 1 */
        const BIITensor *_y; /**< Source tensor 2 */
        BIITensor *_output; /**< Destination tensor */
        bool _has_same_rank; /**< Flag that indicates if condition tensor and other inputs have the same rank */
    };
}
