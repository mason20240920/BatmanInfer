//
// Created by Mason on 2025/4/10.
//

#pragma once
#include <data/core/bi_types.hpp>


namespace BatmanInfer {
    class BIWindow;
    class BIITensor;

    namespace cpu {
        void reduce_RedOpX_reduceX_S32_4(const BIWindow &window,
                                         const BIITensor *input,
                                         BIITensor *output,
                                         const BIReductionOperation op);

        void reduce_RedOpYZW_reduceY_S32_4(const BIWindow &window,
                                           const BIITensor *input,
                                           BIITensor *output,
                                           const BIReductionOperation op);

        void reduce_RedOpYZW_reduceZ_S32_4(const BIWindow &window,
                                           const BIITensor *input,
                                           BIITensor *output,
                                           const BIReductionOperation op);

        void reduce_RedOpYZW_reduceW_S32_4(const BIWindow &window,
                                           const BIITensor *input,
                                           BIITensor *output,
                                           const BIReductionOperation op);
    }
}
