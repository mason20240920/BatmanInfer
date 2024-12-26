//
// Created by Mason on 2024/12/26.
//

#ifndef BATMANINFER_BI_HELPERS_HPP
#define BATMANINFER_BI_HELPERS_HPP

#include <data/core/bi_types.hpp>

namespace BatmanInfer {
    /**
     * @brief 获取给定维度的索引。
     * @param data_layout_dimension 请求该索引的维度。
     * @return
     */
    inline size_t get_data_layout_dimension_index(const BIDataLayoutDimension &data_layout_dimension);

    /** Returns a static map used to find an index or dimension based on a data layout
  *
  * *** Layouts ***
  *
  * *** 4D ***
  * [N C H W]
  * [3 2 1 0]
  * [N H W C]
  *
  * * *** 5D ***
  * [N C D H W]
  * [4 3 2 1 0]
  * [N D H W C]
  */
    const std::vector<BIDataLayoutDimension> &get_layout_map();
}

#include <data/core/bi_helpers.inl>
#endif //BATMANINFER_BI_HELPERS_HPP
