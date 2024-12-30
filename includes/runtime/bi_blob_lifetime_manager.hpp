//
// Created by Mason on 2024/12/27.
//

#ifndef BATMANINFER_BI_BLOB_LIFETIME_MANAGER_HPP
#define BATMANINFER_BI_BLOB_LIFETIME_MANAGER_HPP

#include <runtime/bi_i_simple_lifetime_manager.hpp>
#include <runtime/bi_types.hpp>

namespace BatmanInfer {
    /**
     * @brief 具体类，用于跟踪已注册张量的生命周期，
     *        并计算系统在blob方面的内存需求
     */
    class BIBlobLifetimeManager : public BIISimpleLifetimeManager {
    public:
        using info_type = std::vector<BIBlobInfo>;

    public:
        /**
         * @brief 构造函数
         */
        BIBlobLifetimeManager();

        BIBlobLifetimeManager(const BIBlobLifetimeManager &) = delete;

        BIBlobLifetimeManager &operator=(const BIBlobLifetimeManager &) = delete;

        BIBlobLifetimeManager(BIBlobLifetimeManager &&) = default;

        BIBlobLifetimeManager &operator=(BIBlobLifetimeManager &&) = default;

        /**
         * @brief 访问内存池内部配置元数据的接口
         * @return
         */
        const info_type &info() const;

        std::unique_ptr<BIIMemoryPool> create_pool(BIIAllocator *allocator) override;
        BIMappingType mapping_type() const override;


    private:
        /**
         * @brief 重写继承方法
         */
        void update_blobs_and_mappings() override;
    private:
        /**
         * @brief 内存块
         */
        std::vector<BIBlobInfo> _blobs;
    };
}

#endif //BATMANINFER_BI_BLOB_LIFETIME_MANAGER_HPP
