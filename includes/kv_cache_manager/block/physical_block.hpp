//
// Created by Mason on 2025/5/26.
//

#pragma once

namespace BatmanInfer {
    struct PhysicalBlock {
        int id; // 内存块标识符
        void *buffer; // 指向实际物理内存的指针
        // uint64_t last_access; // 后期LRU访问
        // uint32_t flags; // 状态标识符
    } __attribute__((aligned(64))); // 对齐Cache Line
}
