#include <iostream>

#include <gtest/gtest.h>

int main(int argc, char ** argv) {
    size_t space = 10;

    // 使用 std::shared_ptr 管理动态数组
    std::shared_ptr<uint8_t> ptr(
            new uint8_t[space](), // 分配动态数组并初始化为 0
            [](const uint8_t *ptr) {   // 自定义删除器
                std::cout << "Deleting dynamic array..." << std::endl;
                delete[] ptr;    // 使用 delete[] 释放内存
            }
    );

    // 使用动态数组
    for (size_t i = 0; i < space; ++i) {
        ptr.get()[i] = static_cast<uint8_t>(i); // 赋值
        std::cout << static_cast<int>(ptr.get()[i]) << " ";
    }
    std::cout << std::endl;

    // 当 ptr 超出作用域时，自动调用自定义删除器释放内存

    return EXIT_SUCCESS;
//    ::testing::InitGoogleTest(&argc, argv);
//    return RUN_ALL_TESTS();
}
