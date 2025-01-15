//
// Created by Mason on 2025/1/15.
//

#include <runtime/neon/functions/bi_ne_split.hpp>

namespace BatmanInfer {
    void BINESplit::run() {
        for (unsigned i = 0; i < _num_outputs; ++i) {
            _slice_functions[i].run();
        }
    }
}