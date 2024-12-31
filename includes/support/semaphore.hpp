//
// Created by Mason on 2024/12/31.
//

#ifndef BATMANINFER_SEMAPHORE_HPP
#define BATMANINFER_SEMAPHORE_HPP

#include <support/mutex.hpp>

#include <condition_variable>

namespace BatmanInfer {
#ifndef NO_MULTI_THREADING
    /**
     * @brief 信号量的类
     */
    class Semaphore {
    public:
        /**
         * @brief 默认构造函数
         * @param value 信号量初始化的值
         */
        Semaphore(int value = 0) : _value(value), _m(), _cv() {

        }

        /**
         * @brief 发送信号
         *
         * 增加信号量的值，并通知等待的线程
         *
         */
        inline void signal() {
            {
                // 加锁以保护对信号量的修改
                std::lock_guard<std::mutex> lock(_m);
                // 增加信号量的值
                ++_value;
            }
            // 通知一个等待的线程
            // 在信号量的上下文中，这通常是合理的，因为一次 signal() 只允许一个线程继续执行（即 _value 减少 1）
            // 避免了不必要的线程竞争，提高了效率
            _cv.notify_one();
        }

        /**
         * @brief 等待信号
         *
         * 如果信号量的值为 0, 则线程会阻塞, 直到有其他线程发送信号
         *
         */
        inline void wait() {
            std::unique_lock<std::mutex> lock(_m);
            // 等待条件变量通知，直到信号量的值大于0
            _cv.wait(lock, [this]() { return _value > 0;});
            --_value;
        }

    private:
        int _value;
        Mutex _m;
        std::condition_variable _cv;
    };
#else
    /**
     * @brief 空的 Semaphore 类
     *
     * 如果没有多线程支持，则提供一个空实现的信号量类
     * 所有方法均为空操作，表示无需真正的同步机制。
     */
    class Semaphore {
    public:
        /**
     * @brief 默认构造函数
     *
     * 初始化信号量的值。
     *
     * @param[in] value 信号量的初始值，默认为 0。
     */
    Semaphore(int value = 0) : _value(value)
    {
        (void)_value; // 避免未使用变量的警告
    }

    /**
     * @brief 发送信号
     *
     * 空操作，无需实际实现。
     */
    inline void signal()
    {
        (void)_value; // 避免未使用变量的警告
    }

    /**
     * @brief 等待信号
     *
     * 空操作，无需实际实现。
     */
    inline void wait()
    {
        (void)_value; // 避免未使用变量的警告
    }


    private:
        // 信号量的值
        int _value;
    }
#endif
}

#endif //BATMANINFER_SEMAPHORE_HPP
