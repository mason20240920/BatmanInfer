//
// Created by Mason on 2025/1/14.
//

#pragma once

namespace BatmanGemm {

    template<unsigned int twidth, unsigned int height, bool sve = false, typename Tin, typename Tout>
    void MergeResults(Tout *out, const Tin *in, int ldc, int y0, int ymax, int x0, int xmax, const Tout *bias,
                      Activation act, bool append);

}