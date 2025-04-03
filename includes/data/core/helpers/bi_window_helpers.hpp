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

    /**
     * @brief Update the maximum window for a given tensor shape and border setting
     * @warning Only for @ref BINEGather ops now
     * @param shape Shape of the tensor space
     * @param steps (Optional) Number of elements processed for each step.
     * @param update_window
     */
    void dynamic_calculate_max_window(const BITensorShape &shape, const BISteps &steps, BIWindow &update_window);

    /**
     * @brief 有0维的计算公式
     * @param shape
     * @param steps
     * @param update_window
     */
    void dynamic_origin_max_window(const BITensorShape &shape, const BISteps &steps, BIWindow &update_window);

    inline void dynamic_origin_max_window(const BIITensorInfo &info,
                                          BIWindow &update_window,
                                          const BISteps &steps = BISteps()) {
        dynamic_origin_max_window(info.tensor_shape(), steps, update_window);
    }

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

    /**
     * @brief Update the maximum window for a given tensor shape and border setting
     * @param info
     * @param update_window
     * @param steps
     */
    inline void dynamic_calculate_max_window(const BIITensorInfo &info,
                                             BIWindow &update_window,
                                             const BISteps &steps = BISteps()) {
        dynamic_calculate_max_window(info.tensor_shape(), steps, update_window);
    }

    /**
     * 计算给定张量形状的压缩窗口或最大窗口
     *
     * 如果张量数据在内存中是连续存储的，则张量可以被解释为一个一维数组，
     * 并且所有维度可以合并到x维度中。
     * 否则，将为给定的张量形状生成最大窗口。
     *
     * @param src 定义输入张量形状的张量信息对象
     * @return 内核可以执行的最大窗口以及首选的分割维度
     */
    std::pair<BIWindow, size_t> calculate_squashed_or_max_window(const BIITensorInfo &src);

    /**
     * 计算给定张量形状的压缩窗口或最大窗口
     *
     * 如果张量数据在内存中是连续存储的，则张量可以被解释为一个一维数组，
     * 并且所有维度可以合并到x维度中。
     * 否则，将为给定的张量形状生成最大窗口。
     * @param src0
     * @param src1
     * @return
     */
    std::pair<BIWindow, size_t> calculate_squashed_or_max_window(const BIITensorInfo &src0, const BIITensorInfo &src1);

    /** Function to compute the shape of output and window for the given inputs
      *
      * @param[in] infos Input tensor information
      *
      * @return A pair of the shape and window
      */
    template<typename... Shapes>
    std::pair<BITensorShape, BIWindow> compute_output_shape_and_window(const Shapes &... shapes) {
        const BITensorShape out_shape = BITensorShape::broadcast_shape(shapes...);
        return std::make_pair(out_shape, calculate_max_window(out_shape));
    }

    /** Calculate the maximum window used by a horizontal kernel for a given tensor shape and border setting
     *
     * @param[in] valid_region Valid region object defining the shape of the tensor space for which the window is created.
     * @param[in] steps        (Optional) Number of elements processed for each step.
     * @param[in] skip_border  (Optional) If true exclude the border region from the window.
     * @param[in] border_size  (Optional) Border size. The border region will be excluded from the window.
     *
     * @return The maximum window the kernel can be executed on.
     */
    BIWindow calculate_max_window_horizontal(const BIValidRegion &valid_region,
                                             const BISteps &steps = BISteps(),
                                             bool skip_border = false,
                                             BIBorderSize border_size = BIBorderSize());

    /** Calculate the maximum window used by a horizontal kernel for a given tensor shape and border setting
     *
     * @param[in] info        Tensor info object defining the shape of the object for which the window is created.
     * @param[in] steps       (Optional) Number of elements processed for each step.
     * @param[in] skip_border (Optional) If true exclude the border region from the window.
     * @param[in] border_size (Optional) Border size.
     *
     * @return The maximum window the kernel can be executed on.
     */
    inline BIWindow calculate_max_window_horizontal(const BIITensorInfo &info,
                                                    const BISteps &steps = BISteps(),
                                                    bool skip_border = false,
                                                    BIBorderSize border_size = BIBorderSize()) {
        return calculate_max_window_horizontal(info.valid_region(), steps, skip_border, border_size);
    }


#endif
}

#endif //BATMANINFER_BI_WINDOW_HELPERS_HPP
