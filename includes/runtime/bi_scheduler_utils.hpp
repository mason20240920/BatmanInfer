//
// Created by Mason on 2025/1/4.
//

#ifndef BATMANINFER_BI_SCHEDULER_UTILS_HPP
#define BATMANINFER_BI_SCHEDULER_UTILS_HPP

#include <cstddef>
#include <utility>

namespace BatmanInfer {
    namespace scheduler_utils {
        /** 给定两个维度和一个最大线程数，计算在线程乘积不超过 `max_threads` 的情况下，
  *  最佳的线程组合。
  *
  *  该算法假设两个维度上的工作计算难度相等。
  *
  *  @returns [m_nthreads, n_nthreads] 一个线程对，表示每个维度上应该使用的线程数量。
  *
  *  解释：
  *  1. 功能：
  *     split_2d 函数的目的是在给定的两个维度（m 和 n）以及最大线程数（max_threads）的约束下，
  *     找到一个最佳的线程分配方案，使得：
  *     - 在线程分配的乘积不超过 max_threads。
  *     - 在两个维度上分配的线程数量尽量合理（均衡）。
  *
  *  2. 参数：
  *     - max_threads: 最大可用线程数。用于限制线程分配的总数量。
  *     - m 和 n: 两个维度的大小，表示需要在这两个维度上进行并行计算的工作量。
  *       例如，如果这是一个矩阵计算问题，m 和 n 可以分别表示矩阵的行数和列数。
  *
  *  3. 返回值：
  *     - 返回一个 std::pair<unsigned, unsigned> 类型的值，即 [m_nthreads, n_nthreads]：
  *       - m_nthreads: 在第一个维度（m）上分配的线程数量。
  *       - n_nthreads: 在第二个维度（n）上分配的线程数量。
  *     - 这两个值的乘积满足 m_nthreads * n_nthreads <= max_threads。
  *
  *  4. 算法假设：
  *     - 假设在两个维度上的计算难度是相等的。因此，分配线程时会尽量使两个维度的线程分配均衡，
  *       而不是偏向某一个维度。
  *     - 这种均衡分配可以避免某些线程过载或空闲，从而提高计算效率。
  *
  *  示例：
  *     假设我们有以下参数：
  *     - max_threads = 16
  *     - m = 8（第一个维度的大小）
  *     - n = 4（第二个维度的大小）
  *
  *     目标是找到一个 [m_nthreads, n_nthreads]，使得：
  *     - m_nthreads * n_nthreads <= 16
  *     - 分配尽量均衡。
  *
  *     可能的结果是：
  *     - [m_nthreads, n_nthreads] = [4, 4]，表示在两个维度上分别分配 4 个线程。
  *
  *  使用场景：
  *     - 矩阵运算：在矩阵的行和列上分配线程。
  *     - 图像处理：在图像的宽度和高度上分配线程。
  *     - 多维数据处理：在多维数据的不同轴上分配线程。
  *
  *  通过这个函数，可以在有限的线程资源下，合理分配线程以最大化并行计算性能。
  */
        std::pair<unsigned, unsigned> split_2d(unsigned max_threads, std::size_t m, std::size_t n);
    }
}

#endif //BATMANINFER_BI_SCHEDULER_UTILS_HPP
