//
// Created by holynova on 25-3-12.
//

#include "../source/graph/mutators/bi_mutator_utils.h"

namespace BatmanInfer {

namespace graph {

    bool is_padding_in_height_or_width(const BIDataLayout &layout, const PaddingList &padding_list)
    {
        if (layout == BIDataLayout::NCHW || layout == BIDataLayout::NHWC)
        {
            const unsigned int height_index = get_dimension_idx(layout, BIDataLayoutDimension::HEIGHT);
            const unsigned int width_index  = get_dimension_idx(layout, BIDataLayoutDimension::WIDTH);

            for (unsigned int i = 0; i < padding_list.size(); ++i)
            {
                if (i != height_index && i != width_index && padding_list[i] != PaddingInfo(0, 0))
                {
                    // if the index is not either height or width, don't fuse
                    return false;
                }
            }

            return true;
        }

        return false;
    }

} // namespace graph

} // namespace BatmanInfer
