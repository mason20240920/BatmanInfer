//
// Created by Mason on 2024/12/27.
//

#ifndef BATMANINFER_BI_BLOB_MEMORY_POOL_HPP
#define BATMANINFER_BI_BLOB_MEMORY_POOL_HPP

#include <runtime/bi_i_memory_pool.hpp>
#include <runtime/bi_i_memory_region.hpp>
#include <runtime/bi_types.hpp>

namespace BatmanInfer {
    class BIIAllocator;

    /**
     * @brief Blob内存池
     */
    class BlobMemoryPool : public BIIMemoryPool {

    private:
        /**
         * @brief 分配
         * @param blob_info
         */
        void allocate_blobs(const std::vector<BIBlobInfo> &blob_info);

    private:
        /**
         * @brief 用于内部分配的分配器
         */
        BIIAllocator *_allocator;
        /**
         * @brief 保存所有内存块的向量
         */
        std::vector<std::unique_ptr<BIIMemoryRegion>> _blobs;

        /**
         * @brief 每个内存块的信息
         */
        std::vector<BIBlobInfo> _blob_info;
    };
}


#endif //BATMANINFER_BI_BLOB_MEMORY_POOL_HPP
