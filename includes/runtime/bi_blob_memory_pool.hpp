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
    class BIBlobMemoryPool : public BIIMemoryPool {
    public:
        /**
         * @brief
         *
         * @note 分配器应该比内存池的生命周期更长。
         *
         * @param allocator 分配器
         * @param blob_info 内存信息
         */
        explicit BIBlobMemoryPool(BIIAllocator *allocator,
                                  std::vector<BIBlobInfo> blob_info);

        ~BIBlobMemoryPool();

        BIBlobMemoryPool(const BIBlobMemoryPool &) = delete;

        BIBlobMemoryPool &operator=(const BIBlobMemoryPool &) = delete;

        BIBlobMemoryPool(BIBlobMemoryPool &&) = default;

        BIBlobMemoryPool &operator=(BIBlobMemoryPool &&) = default;

        // 继承的方法重写:
        void acquire(BatmanInfer::BIMemoryMappings &handles) override;
        void release(BatmanInfer::BIMemoryMappings &handles) override;
        BIMappingType mapping_type() const override;
        std::unique_ptr<BIIMemoryPool> duplicate() override;

    private:
        /**
         * @brief 分配内在的内存块
         * @param blob_info
         */
        void allocate_blobs(const std::vector<BIBlobInfo> &blob_info);

        /**
         * @brief 释放内存块
         */
        void free_blobs();

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
