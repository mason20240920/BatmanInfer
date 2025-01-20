//
// Created by Mason on 2025/1/20.
//

#pragma once

#include <data/core/bi_i_tensor_pack.hpp>
#include <function_info/bi_GEMMInfo.h>
#include <runtime/neon/bi_i_ne_operator.hpp>

#include <memory>

namespace BatmanInfer {
    class BIITensor;

    class BIITensorInfo;

    namespace experimental {
        namespace op {
            /**
             * A shallow wrapper for arm_compute::cpu::CpuGemmLowpMatrixMultiplyCore.
             * Any new features should be added to arm_compute::cpu::CpuGemmLowpMatrixMultiplyCore and
             * arm_compute::experimental::op::CpuGEMMLowp should remain a shallow wrapper.
             */
            class BICpuGEMMLowp : public BIINEOperator {
            public:
                /** Constructor */
                BICpuGEMMLowp();

                /** Prevent instances of this class from being copied (As this class contains pointers) */
                BICpuGEMMLowp(const BICpuGEMMLowp &) = delete;

                /** Default move constructor */
                BICpuGEMMLowp(BICpuGEMMLowp &&) = default;

                /** Prevent instances of this class from being copied (As this class contains pointers) */
                BICpuGEMMLowp &operator=(const BICpuGEMMLowp &) = delete;

                /** Default move assignment operator */
                BICpuGEMMLowp &operator=(BICpuGEMMLowp &&) = default;

                /** Default destructor */
                ~BICpuGEMMLowp();

                /** Initialise the kernel's inputs, output
                 *
                 *valid configurations can be referenced in @ref arm_compute::NEGEMMLowpMatrixMultiplyCore.
                 */
                void configure(const BIITensorInfo *a,
                               const BIITensorInfo *b,
                               const BIITensorInfo *c,
                               BIITensorInfo *output,
                               const GEMMInfo &gemm_info = GEMMInfo());

                /** Static function to check if given info will lead to a valid configuration of @ref CpuGEMMLowp
                 *
                 * Similar to @ref CpuGEMMLowp::configure()
                 *
                 * @return a status
                 */
                static BIStatus validate(const BIITensorInfo *a,
                                         const BIITensorInfo *b,
                                         const BIITensorInfo *c,
                                         const BIITensorInfo *output,
                                         const GEMMInfo &gemm_info = GEMMInfo());

                // Inherited methods overridden
                void run(BIITensorPack &tensors) override;

                void prepare(BIITensorPack &tensors) override;

                experimental::BIMemoryRequirements workspace() const override;

            private:
                struct Impl;
                std::unique_ptr<Impl> _impl;
            };
        }
    }
}