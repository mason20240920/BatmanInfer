//
// Created by Mason on 2025/1/3.
//

#ifndef BATMANINFER_BI_I_RUNTIME_CONTEXT_HPP
#define BATMANINFER_BI_I_RUNTIME_CONTEXT_HPP

namespace BatmanInfer {
    // 前向声明
    class BIIScheduler;
    class BIIAssetManager;

    /**
     * @brief 上下文接口
     */
    class BIIRuntimeContext {
    public:
        virtual ~BIIRuntimeContext() = default;

        /**
         * @brief 调度器访问器
         *
         * @note 调度器用于调度工作负载
         *
         * @return 注册到上下文中的调度器
         */
        virtual BIIScheduler *scheduler() = 0;


        /**
         * @brief 资产管理器访问器
         *
         * @note 资产管理器用于管理函数中的对象或张量
         *
         * @return 注册到上下文中的资产管理器
         */
        virtual BIIAssetManager *asset_manager() = 0;
    };
}

#endif //BATMANINFER_BI_I_RUNTIME_CONTEXT_HPP
