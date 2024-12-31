//
// Created by Mason on 2024/12/30.
//

#ifndef BATMANINFER_MUTEX_HPP
#define BATMANINFER_MUTEX_HPP

#include <mutex>

namespace BatmanInfer {
#ifndef NO_MULTI_THREADING
    /**
     * @brief 互斥数据对象的包装器
     */
    using Mutex = std::mutex;

    /**
     * @brief 封装锁对象
     */
    template <typename Mutex>
    using lock_guard = std::lock_guard<Mutex>;

    template <typename Mutex>
    using unique_lock = std::unique_ptr<Mutex>;

#else
    class Mutex {
    public:
        Mutex() = default;

        ~Mutex() = default;

        void lock(){}

        void unlock(){}

        bool try_lock() {
            return true;
        }
    };

    /**
     *  Wrapper implementation of lock-guard data-object
     */
    template <typename Mutex>
    class lock_guard {
    public:
        typedef Mutex mutex_type;

    public:
        explicit lock_guard(Mutex &m_) : m(m_)
        {

        }

        ~lock_guard() {

        }
        lock_guard(const lock_guard &) = delete;

    private:
        mutex_type &m;
    };

    template <typename Mutex>
    class unique_lock {

    public:
        unique_lock() noexcept : m(nullptr) {

        }
        unique_lock(const unique_lock &) = delete;
        unique_lock(unique_lock &&) = default;
        unique_lock &operator=(const unique_lock &) = delete;
        unique_lock &operator=(unique_lock &&) = default;
        ~unique_lock() = default;

        void lock() {

        }

        bool try_lock() {
            return true;
        }

        void unlock() {

        }


    private:
        mutex_type *m;
    };
#endif

}

#endif //BATMANINFER_MUTEX_HPP
