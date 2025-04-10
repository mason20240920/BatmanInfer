//
// Created by holynova on 25-4-8.
//

namespace BatmanInfer {

#ifndef DOXYGEN_SKIP_THIS

    inline float32x4x2_t vmax2q_f32(float32x4x2_t a, float32x4x2_t b)
    {
        float32x4x2_t res = {{vmaxq_f32(a.val[0], b.val[0]), vmaxq_f32(a.val[1], b.val[1])}};
        return res;
    }
#endif /* DOXYGEN_SKIP_THIS */

} // namespace BatmanInfer
