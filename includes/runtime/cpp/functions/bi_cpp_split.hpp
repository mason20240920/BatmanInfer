//
// Created by Mason on 2025/1/15.
//

#pragma once

#include <data/core/bi_error.h>
#include <data/core/bi_helpers.hpp>
#include <data/core/bi_tensor_info.hpp>
#include <data/core/bi_types.hpp>
#include <data/core/utils/misc/bi_shape_calculator.hpp>
#include <runtime/bi_i_function.hpp>

namespace BatmanInfer {
    /**
     * 根据给定的坐标轴切分基本的函数
     * @tparam SliceType
     * @tparam TensorInterfaceType
     */
    template<typename SliceType, typename TensorInterfaceType = BIITensor>
    class BICPPSplit : public BIIFunction {
    public:
        BICPPSplit() : _outputs_vector(), _slice_functions(), _num_outputs(0) {

        }

        /**
         * 静态方法: 验证给定的信息是否会导致 @ref BICPPSplit 有合理配置
         * @param input 输入的向量信息。数据类型: all
         * @param outputs 输出的向量数组: 支持的数据类型: @p input
         * @param axis
         * @return
         */
        static BIStatus validate(const BIITensorInfo *input,
                                 const std::vector<BIITensorInfo *> &outputs,
                                 unsigned int axis) {
            BI_COMPUTE_RETURN_ERROR_ON_NULLPTR(input);
            BI_COMPUTE_RETURN_ERROR_ON(axis >= input->num_dimensions());
            BI_COMPUTE_RETURN_ERROR_ON(outputs.size() < 2);

            // 获取输出的形状
            BITensorShape output_shape{};
            // 输出的形状
            unsigned int total_output_shape_size = 0;

            // 将输出大小求和，如果任何大小为零，则回退到均匀大小的分割。
            const bool using_split_shapes = std::none_of(outputs.begin(), outputs.end(),
                                                         [&total_output_shape_size](BIITensorInfo *info) {
                                                             unsigned int output_shape_size =
                                                                     info->tensor_shape().total_size();
                                                             total_output_shape_size += output_shape_size;
                                                             return output_shape_size == 0;
                                                         });

            if (using_split_shapes)
                BI_COMPUTE_RETURN_ERROR_ON(input->tensor_shape().total_size() != total_output_shape_size);
            else {
                output_shape = BatmanInfer::misc::shape_calculator::compute_split_shape(input, axis, outputs.size());
                BI_COMPUTE_RETURN_ERROR_ON(output_shape.total_size() == 0);
            }

            // 合理的输出张量
            unsigned int axis_offset = 0;
            for (const auto &output: outputs) {
                BI_COMPUTE_RETURN_ERROR_ON_NULLPTR(output);
                if (using_split_shapes) {
                    output_shape = output->tensor_shape();
                    BI_COMPUTE_RETURN_ERROR_ON(output_shape.total_size() == 0);
                }

                const size_t axis_split_step = output_shape[axis];

                // Start/End coordinates
                BICoordinates start_coords;
                BICoordinates end_coords;
                for (unsigned int d = 0; d < output_shape.num_dimensions(); ++d) {
                    end_coords.set(d, -1);
                }

                // Output auto initialization if not yet initialized
                BITensorInfo tmp_output_info = *output->clone();
                if (tmp_output_info.tensor_shape().total_size() == 0)
                    tmp_output_info = input->clone()->set_is_resizable(true).set_tensor_shape(output_shape);


                // Update coordinate on axis
                start_coords.set(axis, axis_offset);
                end_coords.set(axis, axis_offset + axis_split_step);

                BI_COMPUTE_RETURN_ON_ERROR(SliceType::validate(input, output, start_coords, end_coords));
                axis_offset += axis_split_step;
            }

            return BIStatus{};
        }

        /**
         * 初始化内核的输入和输出
         * @param input
         * @param outputs
         * @param axis
         */
        void configure(const TensorInterfaceType *input,
                       const std::vector<TensorInterfaceType *> &outputs,
                       unsigned int axis) {
            // 创建切分函数
            _num_outputs = outputs.size();
            _slice_functions.resize(_num_outputs);

            // 导出输出张量信息
            std::vector<BIITensorInfo *> outputs_info;
            for (auto &output: outputs) {
                BI_COMPUTE_ERROR_ON_NULLPTR(output);
                outputs_info.emplace_back(output->info());
            }

            // 如果任何输出都有0的大小，返回使用平均大小输出
            const bool outputs_have_sizes = std::none_of(outputs_info.begin(), outputs_info.end(),
                                                         [](BIITensorInfo *info) {
                                                             return info->tensor_shape().total_size() == 0;
                                                         });

            // 验证是否合法
            BI_COMPUTE_ERROR_THROW_ON(BICPPSplit::validate(input->info(), outputs_info, axis));

            unsigned int axis_offset = 0;
            unsigned int i = 0;

            for (const auto &output_info: outputs_info) {
                // 获取输出形状
                BITensorShape output_shape = (outputs_have_sizes ? output_info->tensor_shape()
                                                                 : BatmanInfer::misc::shape_calculator::compute_split_shape(
                                input->info(), axis, _num_outputs));

                const size_t axis_split_step = output_shape[axis];

                // 开始/结束的 坐标
                BICoordinates start_coords;
                BICoordinates end_coords;

                for (unsigned int d = 0; d < output_shape.num_dimensions(); ++d) {
                    end_coords.set(d, -1);
                }

                // Update coordinate on axis
                start_coords.set(axis, axis_offset);
                end_coords.set(axis, axis_offset + axis_split_step);

                // Configure slice function
                _slice_functions[i].configure(input, outputs[i], start_coords, end_coords);

                // Set valid region from shape
                outputs[i]->info()->set_valid_region(BIValidRegion(BICoordinates(), output_shape));

                // Update axis offset
                axis_offset += axis_split_step;
                ++i;
            }
        }


    protected:
        // 输出向量组
        std::vector<TensorInterfaceType *> _outputs_vector;
        // 输出的函数方法
        std::vector<SliceType> _slice_functions;
        // 输出的数量
        unsigned int _num_outputs;
    };
}