//
// Created by Mason on 2024/12/31.
//

#include <data/core/bi_i_tensor.hpp>
#include <data/core/bi_window.hpp>

namespace BatmanInfer {
    void BIITensor::copy_from(const BatmanInfer::BIITensor &src) {
        if (&src == this)
            return;

        const BIITensorInfo *src_info = src.info();
        BIITensorInfo       *dst_info = this->info();

        // 目标张量维度小于源维度
        BI_COMPUTE_ERROR_ON(src_info->num_dimensions() > dst_info->num_dimensions());
        BI_COMPUTE_ERROR_ON(src_info->num_channels() != dst_info->num_channels());
        BI_COMPUTE_ERROR_ON(src_info->element_size() != dst_info->element_size());

        for (size_t d = 0; d < src_info->num_dimensions(); d++)
            BI_COMPUTE_ERROR_ON(src_info->dimension(d) > dst_info->dimension(d));

        // 拷贝关于可用区间的信息
        dst_info->set_valid_region(src_info->valid_region());

        BIWindow win_src;
        win_src.use_tensor_dimensions(src_info->tensor_shape(), BIWindow::DimY);
        BIWindow win_dst;
        win_dst.use_tensor_dimensions(dst_info->tensor_shape(), BIWindow::DimY);


    }
}