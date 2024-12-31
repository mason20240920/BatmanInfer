//
// Created by Mason on 2024/12/27.
//

#ifndef BATMANINFER_RUNTIME_BI_TYPES_HPP
#define BATMANINFER_RUNTIME_BI_TYPES_HPP

#include <runtime/bi_i_memory.hpp>

#include <map>

namespace BatmanInfer {
    enum class BIMappingType {
        /**
         * @brief 映射是以大块粒度进行的。
         */
        BLOBS,
        /**
         * @brief 映射在同一数据块中以偏移粒度存在。
         */
        OFFSETS,
    };

    /** (handle，索引/偏移) 的映射，其中handle是对象的内存handle
     * 用于提供内存，索引/偏移是应使用的池中的缓冲区/偏移量
     * @note 所有对象都预先固定到特定的缓冲区，以避免任何相关的开销
     * */
    using BIMemoryMappings = std::map<BIIMemory *, size_t>;

    /**
     * @brief 一个关于组和内存映射的地图
     */
    using BIGroupMappings = std::map<size_t, BIMemoryMappings >;

    /**
     * @brief  Blob（数据块） 的元数据信息
     */
    struct BIBlobInfo {

        explicit BIBlobInfo(size_t size_ = 0,
                   size_t alignment_ = 0,
                   size_t owners_ = 1) :
                   size(size_),
                   alignment(alignment_),
                   owners(owners_) {

        }

        /**
         * @brief 数据块大小
         */
        size_t size;

        /**
         * @brief 数据块的对齐
         */
        size_t alignment;
        /**
         * @brief 共享该 Blob 的对象数量
         */
        size_t owners;
    };


}

#endif //BATMANINFER_RUNTIME_BI_TYPES_HPP
