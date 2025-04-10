//
// Created by Mason on 2025/4/10.
//

#pragma once
#include <data/core/bi_types.hpp>


namespace BatmanInfer {
    class BIWindow;
    class BIITensor;

    namespace cpu {
        void reduce_RedOpX_reduceX_qasymm8(const BIWindow &window,
                                           const BIITensor *input,
                                           BIITensor *output,
                                           const BIReductionOperation op);

        void reduce_RedOpYZW_reduceY_qasymm8(const BIWindow &window,
                                             const BIITensor *input,
                                             BIITensor *output,
                                             const BIReductionOperation op);

        void reduce_RedOpYZW_reduceZ_qasymm8(const BIWindow &window,
                                             const BIITensor *input,
                                             BIITensor *output,
                                             const BIReductionOperation op);

        void reduce_RedOpYZW_reduceW_qasymm8(const BIWindow &window,
                                             const BIITensor *input,
                                             BIITensor *output,
                                             const BIReductionOperation op);
    }
}
