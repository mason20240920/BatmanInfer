#include <iostream>

#include <gtest/gtest.h>

/**
 * @brief 模拟内存管理类
 */
class BIIMemory {
public:
    void *allocate(size_t size, size_t alignment) {
        void *ptr = nullptr;
        posix_memalign(&ptr, alignment, size);
        return ptr;
    }

    void deallocate(void *ptr) {
        free(ptr);
    }
};

class ObjectManager {
public:
    virtual void end_life_time(void *obj,
                               BIIMemory &obj_memory,
                               size_t size,
                               size_t alignment) = 0;
};

class MemoryManager : public ObjectManager {
public:
    void end_life_time(void *obj, BIIMemory &obj_memory, size_t size, size_t alignment) override {
        if (obj) {
            std::cout << "Cleaning up object memory..." << std::endl;
            std::memset(obj, 0, size);
        }

        obj_memory.deallocate(obj);
        std::cout << "Memory of size " << size << " bytes freed." << std::endl;
    }
};

int main(int argc, char ** argv) {
    BIIMemory memory;
    MemoryManager my_memory_manager;

    size_t size = 64;         // 对象大小
    size_t alignment = 8;     // 对齐要求
    void *obj = memory.allocate(size, alignment);

    // 初始化对象
    std::memset(obj, 42, size); // 将对象内存填充为42
    std::cout << "Object created and initialized." << std::endl;

    // 打印对象的部分数据
    std::cout << "Object data (first byte): " << *((char *)obj) << std::endl;

    my_memory_manager.end_life_time(obj, memory, size, alignment);

    return EXIT_SUCCESS;
//    ::testing::InitGoogleTest(&argc, argv);
//    return RUN_ALL_TESTS();
}
