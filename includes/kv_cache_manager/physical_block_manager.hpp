//
// Created by Mason on 2025/5/26.
//

#pragma once
#include <atomic>
#include <stdlib.h>
#include <kv_cache_manager/block/physical_block.hpp>

#include "data/core/bi_error.h"

namespace BatmanInfer {
    namespace {
        // 跨平台对齐内存分配函数
        void *platform_aligned_alloc(size_t alignment, size_t size) {
#ifdef __ANDROID__
            // Android 使用 posix_memalign
            void* ptr = nullptr;
            if (posix_memalign(&ptr, alignment, size) != 0) {
                return nullptr;  // 分配失败
            }
            return ptr;
#else
            // 其他平台使用 aligned_alloc
            // aligned_alloc 要求 size 是 alignment 的倍数
            size_t aligned_size = ((size + alignment - 1) / alignment) * alignment;
            return aligned_alloc(alignment, aligned_size);
#endif
        }
    }

    /**
     * @brief 内存块管理器
     */
    struct PhysicalBlockManager {
        PhysicalBlock *blocks; // 物理块组
        size_t num_blocks; // 块数量
        size_t block_size; // 每个块的大小(bytes)

        // 状态与统计
        std::atomic<int> free_stack_top; // 栈顶指针
        int *free_stack; // 空闲块索引栈
        std::atomic<int> free_blocks; // 可用块总数

        // 内存管理器
        void *memory_pool; // 预分配内存
    };

    /**
     * @brief 自动分配内存ID组, 返回值是多个包含块指针和分配的结构
     */
    struct MultiBlockAllocation {
        PhysicalBlock **blocks; // 指向块指针的指针数组
        int *block_ids; // 块id数组
        size_t count; // 实际获取块数量

        ~MultiBlockAllocation() {
            delete[] block_ids;
            delete[] blocks;
        }
    };


    inline PhysicalBlockManager *create_block_manager(size_t num_blocks, size_t block_size) {
        constexpr size_t aligned_size = ((sizeof(PhysicalBlockManager) + 63) / 64) * 64;
        auto manager = static_cast<PhysicalBlockManager *>(platform_aligned_alloc(64, aligned_size));
        if (!manager) {
            BI_COMPUTE_ERROR("Failed to allocate memory for physical block manager");
        }

        manager->memory_pool = platform_aligned_alloc(4096, num_blocks * block_size);
        if (!manager->memory_pool) {
            free(manager);
            return nullptr;
        }
        // #endif

        // 对齐的内存block_size
        constexpr size_t aligned_block_size = ((sizeof(PhysicalBlock) + 63) / 64) * 64;
        // 分配块数组, 确保每个块都对齐到缓存行
        manager->blocks = static_cast<PhysicalBlock *>(platform_aligned_alloc(64, aligned_block_size * num_blocks));
        if (!manager->blocks) {
            free(manager->memory_pool);
            free(manager);
            return nullptr;
        }

        // 初始化每个块 => 每个块使用其索引作为固定ID
        for (size_t i = 0; i < num_blocks; i++) {
            manager->blocks[i].id = i;
            manager->blocks[i].buffer = static_cast<char *>(manager->memory_pool) + (i * block_size);
        }

        manager->free_stack = static_cast<int *>(platform_aligned_alloc(64, num_blocks * sizeof(int)));
        if (!manager->free_stack) {
            free(manager->memory_pool);
            free(manager->blocks);
            free(manager);
            return nullptr;
        }

        // 分配和初始化空闲块栈 - 所有块开始时都是空闲的
        for (size_t i = 0; i < num_blocks; i++) {
            // 可以考虑不同的初始化策略，这里使用顺序编号
            manager->free_stack[i] = static_cast<int>(i);
        }
        manager->num_blocks = num_blocks;

        // 原子操作不会强制任何内存访问的顺序约束
        manager->free_stack_top.store(static_cast<int>(num_blocks) - 1, std::memory_order_relaxed);
        manager->free_blocks.store(static_cast<int>(num_blocks), std::memory_order_relaxed);

        return manager;
    }

    /**
     * @brief 重置管理器里面KV Cache的内存块, 保留root_id
     * @param manager
     */
    inline void reset_block_manager(PhysicalBlockManager *manager) {
        for (size_t i = 0; i < (manager->num_blocks - 1); i++) {
            manager->free_stack[i] = static_cast<int>(i);
        }
        manager->free_stack_top.store(static_cast<int>(manager->num_blocks) - 2, std::memory_order_relaxed);
        manager->free_blocks.store(static_cast<int>(manager->num_blocks - 1), std::memory_order_relaxed);
    }

    /**
     * @brief 申请物理内存块
     * @param manager
     * @param request_count: 请求内存数量
     * @return
     */
    inline MultiBlockAllocation acquire_block(PhysicalBlockManager *manager,
                                              const size_t request_count) {
        // 边界检查(查看申请的block数量是不是大于0)
        if (request_count == 0)
            return {nullptr, nullptr, 0};

        // 预分配结果数组
        auto *result_blocks = new PhysicalBlock *[request_count];
        auto *result_ids = new int[request_count];
        size_t acquired = 0;

        // 尝试从空闲栈中获取一个块
        int top = manager->free_stack_top.load(std::memory_order_acquire);
        while (top >= 0 && acquired < request_count) {
            // 计算可获取的块数量(当前栈深度和请求数量的较小值)
            const size_t available = std::min<size_t>(top + 1, request_count - acquired);
            const int new_top = top - available;
            // 尝试弹出栈顶元素
            if (manager->free_stack_top.compare_exchange_weak(top, new_top, std::memory_order_acq_rel)) {
                // 优化2: 批量获取预留的块
                for (size_t i = 0; i < available; ++i) {
                    const int block_idx = manager->free_stack[top - i];
                    PhysicalBlock *block = &manager->blocks[block_idx]; // 物理block块


                    result_blocks[acquired] = block;
                    result_ids[acquired] = block->id;

                    // 优化3: 预取技术 - 提前加载内存以减少缓存缺失
                    __builtin_prefetch(block->buffer, 1, 3);

                    acquired++;
                }

                // 优化4: 一次性减少空闲块计数
                if (acquired > 0) {
                    manager->free_blocks.fetch_sub(static_cast<int>(acquired), std::memory_order_release);
                }

                // 如果已获取所需数量的块，跳出循环
                if (acquired == request_count) {
                    break;
                }
            }

            // CAS失败，重新加载栈顶
            top = manager->free_stack_top.load(std::memory_order_acquire);
        }

        // 如果一个块都没获取到, 释放资源
        if (acquired == 0) {
            delete[] result_blocks;
            delete[] result_ids;
            return {nullptr, nullptr, 0};
        }

        if (acquired != request_count) {
            delete[] result_blocks;
            delete[] result_ids;
            throw std::runtime_error("Failed to acquire free blocks.");
        }

        return {result_blocks, result_ids, acquired};
    }

    /**
     * @brief 根据块内存进行释放block
     * @param manager: 块管理器
     * @param block_ids 块的组ids
     * @param count 块的数量
     */
    inline void release_blocks(PhysicalBlockManager *manager, const int *block_ids, const size_t count) {
        if (count == 0) return;

        // 阶段1: 批量回收需要释放的块
        // 分组回收以减少自旋锁争用
        constexpr size_t BATCH_SIZE = 16;
        const size_t batches = (count + BATCH_SIZE - 1) / BATCH_SIZE;

        // 一次性更新空闲块计数 - 避免多次原子操作
        manager->free_blocks.fetch_add(static_cast<int>(count), std::memory_order_release);

        int recycled = 0;
        for (size_t batch = 0; batch < batches; batch++) {
            // 计算本批次需要回收的块数量
            size_t current_batch_size = 0;
            int batch_block_ids[BATCH_SIZE];

            const size_t start_idx = batch * BATCH_SIZE;
            const size_t end_idx = std::min(start_idx + BATCH_SIZE, count);
            for (size_t i = start_idx; i < end_idx; ++i) {
                batch_block_ids[current_batch_size++] = block_ids[i];
                recycled++;
            }

            // 批量添加到空闲栈
            if (current_batch_size > 0) {
                // 获取并更新栈顶
                int top = manager->free_stack_top.load(std::memory_order_acquire);
                bool success = false;

                // 自适应退避 - 使用指数退避减少争用
                int backoff = 1;
                do {
                    // 准备新的栈状态
                    for (size_t i = 0; i < current_batch_size; ++i) {
                        manager->free_stack[top + 1 + i] = batch_block_ids[i];
                    }

                    // 尝试批量更新栈顶
                    success = manager->free_stack_top.compare_exchange_weak(
                        top, top + current_batch_size, std::memory_order_acq_rel);

                    if (!success) {
                        // 退避以减少自旋浪费
                        for (int i = 0; i < backoff; ++i) {
                            // 通用实现：空操作循环
                            for (volatile int j = 0; j < 50; ++j) {
                            }
                        }
                        backoff = std::min(backoff * 2, 1024); // 指数退避，上限1024
                    }
                } while (!success);
            }
        }
    }

    // 兼容单块释放的接口
    inline void release_block(PhysicalBlockManager *manager, const int block_id) {
        release_blocks(manager, &block_id, 1);
    }
}
