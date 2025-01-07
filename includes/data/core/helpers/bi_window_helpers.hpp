//
// Created by Mason on 2025/1/7.
//

#ifndef BATMANINFER_BI_WINDOW_HELPERS_HPP
#define BATMANINFER_BI_WINDOW_HELPERS_HPP

#include <data/core/bi_window.hpp>
#include <data/core/bi_steps.hpp>

namespace BatmanInfer {
#ifndef DOXYGEN_SKIP_THIS

    /**
     *
     * 为给定的张量形状和边界设置计算最大窗口
     *
     * @param valid_region 有效区域对象，定义了用于创建窗口的张量空间的形状。
     * @param steps （可选）每次步进处理的元素数量。
     * @param skip_border （可选）如果为 true，则从窗口中排除边界区域。
     * @param border_size （可选）边界大小。
     * @return
     */
    BIWindow calculate_max_window(const BIValidRegion &valid_region,
                                  const BISteps &steps = BISteps(),
                                  bool skip_border = false,
                                  BIBorderSize border_size = BIBorderSize());

    /** Calculate the maximum window for a given tensor shape and border setting
     *
     * @param[in] shape       Shape of the tensor space
     * @param[in] steps       (Optional) Number of elements processed for each step.
     * @param[in] skip_border (Optional) If true exclude the border region from the window.
     * @param[in] border_size (Optional) Border size.
     *
     * @return The maximum window the kernel can be executed on.
     */
    BIWindow calculate_max_window(const BITensorShape &shape,
                                  const BISteps &steps = BISteps(),
                                  bool skip_border = false,
                                  BIBorderSize border_size = BIBorderSize());

    /** Calculate the maximum window for a given tensor shape and border setting
     *
     * @param[in] info        Tensor info object defining the shape of the object for which the window is created.
     * @param[in] steps       (Optional) Number of elements processed for each step.
     * @param[in] skip_border (Optional) If true exclude the border region from the window.
     * @param[in] border_size (Optional) Border size.
     *
     * @return The maximum window the kernel can be executed on.
     */
    inline BIWindow calculate_max_window(const BIITensorInfo &info,
                                         const BISteps &steps = BISteps(),
                                         bool skip_border = false,
                                         BIBorderSize border_size = BIBorderSize()) {
        return calculate_max_window(info.tensor_shape(), steps, skip_border, border_size);
    }

#endif
}

#endif //BATMANINFER_BI_WINDOW_HELPERS_HPP
