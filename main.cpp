#include <iostream>

#include <gtest/gtest.h>

#include <thread>
#include <semaphore> // C++20 信号量

std::counting_semaphore<10> parking_lot(10); // 初始化信号量为 5，表示有 5 个车位

void driver(int id) {
    std::cout << "Driver " << id << " is looking for a parking spot...\n";

    // 请求车位（wait）
    parking_lot.acquire();
    std::cout << "Driver " << id << " has parked the car.\n";

    // 模拟停车一段时间
    std::this_thread::sleep_for(std::chrono::seconds(2));

    // 释放车位（signal）
    std::cout << "Driver " << id << " is leaving the parking lot.\n";
    parking_lot.release();
}

int main(int argc, char ** argv) {
    // 创建 10 个司机线程
    std::thread drivers[10];
    for (int i = 0; i < 10; ++i) {
        drivers[i] = std::thread(driver, i + 1);
    }

    // 等待所有线程完成
    for (int i = 0; i < 10; ++i) {
        drivers[i].join();
    }

    return EXIT_SUCCESS;
//    ::testing::InitGoogleTest(&argc, argv);
//    return RUN_ALL_TESTS();
}
