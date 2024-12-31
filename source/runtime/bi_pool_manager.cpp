//
// Created by Mason on 2024/12/31.
//

#include <runtime/bi_pool_manager.hpp>

#include <data/core/bi_error.h>
#include <runtime/bi_i_memory_pool.hpp>

using namespace BatmanInfer;

BIPoolManager::BIPoolManager() : _free_pools(), _occupied_pools(), _sem(), _mtx()
{

}

BIIMemoryPool *BIPoolManager::lock_pool() {
    // 没有设置内存池
    BI_COMPUTE_ERROR_ON_MSG(_free_pools.empty() && _occupied_pools.empty(), "Haven't setup any pools!");

    // 使用信号量的 wait() 方法阻塞当前线程，直到有可用的内存池资源。
    // 信号量的值代表可用内存池的数量，如果值为 0，则线程会阻塞。
    _sem->wait();

    // 加锁保护共享资源（_free_pools 和 _occupied_pools）的修改，确保线程安全。
    lock_guard<Mutex> lock(_mtx);

    // 再次检查 _free_pools 是否为空。
    // 如果为空，则抛出错误，因为信号量已经表明至少有一个可用的内存池。
    BI_COMPUTE_ERROR_ON_MSG(_free_pools.empty(), "Empty pool must exist as semaphore has been signalled");

    // 从 _free_pools 中取出一个内存池，移动到 _occupied_pools 中。
    // `splice` 是一种高效的操作，它将 _free_pools 的第一个元素移动到 _occupied_pools 的开头。
    _occupied_pools.splice(std::begin(_occupied_pools), _free_pools, std::begin(_free_pools));

    // 返回刚刚移动到 _occupied_pools 的内存池对象的指针。
    return _occupied_pools.front().get();
}

void BIPoolManager::unlock_pool(BatmanInfer::BIIMemoryPool *pool) {
    BI_COMPUTE_ERROR_ON_MSG(_free_pools.empty() && _occupied_pools.empty(), "Haven't setup any pools!");

    lock_guard<Mutex> lock(_mtx);
    // 找到线程池所在的迭代器
    auto it = std::find_if(std::begin(_occupied_pools), std::end(_occupied_pools),
                           [pool](const std::unique_ptr<BIIMemoryPool> &pool_it) { return pool_it.get() == pool; });
    BI_COMPUTE_ERROR_ON_MSG(it == std::end(_occupied_pools), "Pool to be unlocked couldn't be found!");
    _free_pools.splice(std::begin(_free_pools), _occupied_pools, it);
    _sem->signal();
}

void BIPoolManager::register_pool(std::unique_ptr<BIIMemoryPool> pool) {
    lock_guard<Mutex> lock(_mtx);
    BI_COMPUTE_ERROR_ON_MSG(!_occupied_pools.empty(), "All pools should be free in order to register a new one!");

    // Set pool
    _free_pools.push_front(std::move(pool));

    // 更新信号量
    _sem = std::make_unique<Semaphore>(_free_pools.size());
}

std::unique_ptr<BIIMemoryPool> BIPoolManager::release_pool() {
    lock_guard<Mutex> lock(_mtx);
    BI_COMPUTE_ERROR_ON_MSG(!_occupied_pools.empty(), "All pools should be free in order to release one!");

    if (!_free_pools.empty())
    {
        std::unique_ptr<BIIMemoryPool> pool = std::move(_free_pools.front());
        BI_COMPUTE_ERROR_ON(_free_pools.front() != nullptr);
        _free_pools.pop_front();

        // Update semaphore
        _sem = std::make_unique<Semaphore>(_free_pools.size());

        return pool;
    }

    return nullptr;
}

void BIPoolManager::clear_pools() {
    lock_guard<Mutex> lock(_mtx);
    BI_COMPUTE_ERROR_ON_MSG(!_occupied_pools.empty(), "All pools should be free in order to clear the PoolManager!");
    _free_pools.clear();

    // Update semaphore
    _sem = nullptr;

    // 作用域结束，lock_guard 自动释放锁
}

size_t BIPoolManager::num_pools() const {
    lock_guard<Mutex> lock(_mtx);

    return _free_pools.size() + _occupied_pools.size();
}