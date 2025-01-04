//
// Created by Mason on 2025/1/4.
//

#include <runtime/bi_scheduler_utils.hpp>

#include <cmath>
#include <algorithm>

namespace BatmanInfer {
    namespace scheduler_utils {
#ifndef BARE_METAL

        std::pair<unsigned, unsigned> split_2d(unsigned max_threads, std::size_t m, std::size_t n) {
            /*
             * 我们希望线程在 M 和 N 维度上的比例与问题规模 m 和 n 的比例相同。
             *
             * 因此：  mt/nt == m/n    其中 mt * nt == max_threads
             *
             *        max_threads/nt = mt    并且    (max_threads/nt) * (m/n) = nt
             *        nt^2 = max_threads * (m/n)
             *        nt = sqrt(max_threads * (m/n))
             */
            // m 和 n 问题维度的比例（m/n）
            double ratio = m / static_cast<double>(n);

            // nt = sqrt(max_threads * (m / n))
            const unsigned adjusted = std::round(std::sqrt(max_threads * ratio));

            // 找到最接近的 max_threads 的因子
            for (unsigned i = 0; i != adjusted; ++i) {
                // 尝试向下调整
                const unsigned adj_down = adjusted - i;
                if (max_threads % adj_down == 0) {
                    return {adj_down, max_threads / adj_down};
                }

                // 尝试向上调整
                const unsigned adj_up = adjusted + i;
                if (max_threads % adj_up == 0) {
                    return {adj_up, max_threads / adj_up};
                }
            }

            // 如果没有找到合适的结果，则根据最大维度进行偏置分配
            if (m > n) {
                return {std::min<unsigned>(m, max_threads), 1};
            } else {
                return {1, std::min<unsigned>(n, max_threads)};
            }
        }

#endif
    }
}