//
// Created by Mason on 2025/5/26.
// GPT-2内核的KV Cache管理器接口

#pragma once

namespace BatmanInfer {
    /**
     * @brief KV Cache管理器
     */
    class IKVCacheManager {
    public:
        virtual ~IKVCacheManager() = default;
    };
}
