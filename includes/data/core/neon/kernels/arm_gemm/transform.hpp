//
// Created by Mason on 2025/1/14.
//

#pragma once

#include "utils.hpp"

namespace BatmanGemm {

    template<unsigned int IntBy, unsigned int BlockBy, bool Transposed, VLType vlt = VLType::None, typename TOut, typename TIn>
    void Transform(
            TOut *out, const TIn *const in, const int stride,
            const int k0, const int kmax, const int x0, const int xmax
    );
}