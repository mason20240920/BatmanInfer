//
// Created by Mason on 2024/12/25.
//

#ifndef BATMANINFER_BI_I_MEMORY_REGION_H
#define BATMANINFER_BI_I_MEMORY_REGION_H

#include <cstddef>
#include <memory>

namespace BatmanInfer {
    /**
     * @brief 内存区域接口
     */
    class BIIMemoryRegion {
    public:
        /**
         * @brief
         * @param size
         */
        explicit BIIMemoryRegion(size_t size): _size(size) {

        }

        /**
         * @brief 默认析构函数
         */
        virtual ~BIIMemoryRegion() = default;

        /**
         * @brief 从内存中提取一个子区域。
         *
         * @warning 所有权由父内存维护，而此函数返回一个包装的原始内存区域。因此，父内存在此之前不应被释放。
         *
         * @param offset 偏移到该区域
         * @param size
         * @return
         */
        virtual std::unique_ptr<BIIMemoryRegion> extract_subregion(size_t offset,
                                                                   size_t size) = 0;

        /**
         * @brief 返回指向分配数据的指针。
         * @return
         */
        virtual void *buffer() = 0;

        /**
         * @brief 返回指向分配数据的指针。
         * @return
         */
        virtual const void *buffer() const = 0;

        /**
         * @brief 内存区域大小访问器
         * @return
         */
        size_t size() const {
            return _size;
        }

        /**
         * @brief
         *
         * @warning This should only be used in correlation with handle
         *
         * @param size
         */
        void set_size(size_t size) {
            _size = size;
        }

    protected:
        size_t _size;
    };
}

#endif //BATMANINFER_BI_I_MEMORY_REGION_H
