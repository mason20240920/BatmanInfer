//
// Created by holynova on 25-3-12.
//

#pragma once

#include "graph/bi_utils.h"

namespace BatmanInfer {

namespace graph {

    /** Check if padding is in height and/or width dimensions
     *
     * @param[in] layout       Data layout of the tensor
     * @param[in] padding_list List of padding pairs
     */
    bool is_padding_in_height_or_width(const BIDataLayout &layout, const PaddingList &padding_list);

} // namespace graph

} // namespace BatmanInfer
