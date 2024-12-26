#include <iostream>

#include <gtest/gtest.h>

// 抽象基类
class Allocator {
public:
    virtual ~Allocator() = default;

    /**
     * @brief 由子类实现的接口，用于分配字节。
     * @param size 分配的大小
     * @param alignment 返回的指针应遵循的对齐方式
     * @return 指向已分配内存的指针
     */
    virtual void* allocate(size_t size, size_t alignment) = 0;

    /**
     * @brief 用于释放内存的接口
     * @param ptr 要释放的内存指针
     */
    virtual void deallocate(void* ptr) = 0;
};

// 子类实现
class AlignedAllocator : public Allocator {
public:
    // 分配内存
    void* allocate(size_t size, size_t alignment) override {
        void* ptr = nullptr;

        // 使用 C++17 的 std::aligned_alloc
        ptr = std::aligned_alloc(alignment, size);

        if (!ptr) {
            throw std::bad_alloc(); // 分配失败，抛出异常
        }

        return ptr;
    }

    // 释放内存
    void deallocate(void* ptr) override {
        std::free(ptr); // 释放内存
    }
};

int main(int argc, char ** argv) {
    AlignedAllocator allocator;

    size_t size = 64;       // 分配 64 字节
    size_t alignment = 16;  // 对齐到 16 字节

    // 分配内存
    void* ptr = allocator.allocate(size, alignment);

    // 检查指针地址是否满足对齐要求
    std::cout << "Allocated memory address: " << ptr << std::endl;
    if (reinterpret_cast<uintptr_t>(ptr) % alignment == 0) {
        std::cout << "Memory is aligned to " << alignment << " bytes." << std::endl;
    } else {
        std::cout << "Memory is NOT aligned!" << std::endl;
    }

    // 使用完毕，释放内存
    allocator.deallocate(ptr);

    return EXIT_SUCCESS;
//    ::testing::InitGoogleTest(&argc, argv);
//    return RUN_ALL_TESTS();
}
