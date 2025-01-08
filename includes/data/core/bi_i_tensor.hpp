//
// Created by Mason on 2024/12/31.
//

#ifndef BATMANINFER_BI_I_TENSOR_HPP
#define BATMANINFER_BI_I_TENSOR_HPP

#include <data/core/bi_i_tensor_info.hpp>

#include <cstdint>

namespace BatmanInfer {
    /**
     * @brief CPU张量的接口
     */
    class BIITensor {
    public:
        /**
         * @brief 接口函数: 返回张量原始数据
         * @return 返回张量接口信息的指针
         */
        virtual BIITensorInfo *info() const = 0;

        /**
         * @brief  接口函数: 返回张量原始数据
         * @return 返回张量接口信息的指针
         */
        virtual BIITensorInfo *info() = 0;

        virtual ~BIITensor() = default;

        /**
         * @brief 返回CPU内存的指针
         * @return
         */
        virtual uint8_t *buffer() const = 0;

        /**
         * @brief 根据坐标点获取元素的指针
         * @param id
         * @return
         */
        inline uint8_t *ptr_to_element(const BICoordinates &id) const {
            return buffer() + info()->offset_element_in_bytes(id);
        }

        /**
         * @brief 复制另一个张量的内容
         *
         * @note 源张量的维度数量必须小于或等于目标张量的维度数量
         *
         * @note 目标张量的所有维度必须大于或等于源张量的对应维度。
         *
         * @note 源张量和目标张量的 num_channels() 和 element_size() 必须一致。
         *
         * @param src
         */
        void copy_from(const BIITensor &src);

#ifdef BI_COMPUTE_ASSERTS_ENABLED

        /**
         * @brief 使用用户定义的格式化信息将张量打印到指定的流中。
         * @param s
         * @param io_fmt
         */
        void print(std::ostream &s,
                   BIIOFormatInfo io_fmt = BIIOFormatInfo()) const;

#endif

        /**
         * @brief 指示位: 表示张量是否使用
         *
         * @return 如果他被使用返回true
         */
        bool is_used() const;

        void mark_as_unused() const;

        void mark_as_used() const;

    private:
        mutable bool _is_used = {true};
    };
}

#endif //BATMANINFER_BI_I_TENSOR_HPP
