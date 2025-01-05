//
// Created by Mason on 2025/1/5.
//

#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <initializer_list>

namespace BatmanGemm {
    template<unsigned int D>
    class BINDRange {
    private:
        std::array<unsigned int, D> m_sizes{};
        std::array<unsigned int, D> m_total_sizes{};

        /**
         * @brief 迭代器
         */
        class BINDRangeIterator {
        private:
            const BINDRange &m_parent;
            unsigned int    m_pos = 0;
            unsigned int    m_end = 0;

        public:
            BINDRangeIterator(const BINDRange &p,
                              unsigned int s,
                              unsigned int e) : m_parent(p),
                                                m_pos(s),
                                                m_end(e) {

            }

            bool done() const {
                return m_pos >= m_end;
            }

            unsigned int dim(unsigned int d) const {
                // 初始化 r 为当前线性索引 m_pos
                unsigned int r = m_pos;

                // 如果 d < D - 1，则对 r 取模，模数为第 d 维的累积大小 m_total_sizes[d]
                if (d < (D - 1))
                    r %= m_parent.m_total_sizes[d];

                // 如果 d > 0，则对 r 进行整除，除数为第 d-1 维的累积大小 m_total_sizes[d-1]
                // 这一步的作用是去掉更低维度的影响，提取当前维度的坐标
                if (d > 0)
                    r /= m_parent.m_total_sizes[d - 1];

                // 返回第 d 维的坐标
                return r;
            }

            bool next_dim0() {
                m_pos++;

                return !done();
            }

            bool next_dim1() {
                m_pos += m_parent.m_sizes[0] - dim(0);

                return !done();
            }

            unsigned int dim0_max() const {
                unsigned int offset = std::min(m_end - m_pos, m_parent.m_sizes[0] - dim(0));

                return dim(0) + offset;
            }
        };

        void set_total_sizes() {
            unsigned int t = 1;

            for (unsigned int i = 0; i < D; i++) {
                if (m_sizes[i] == 0) {
                    m_sizes[i] = 1;
                }

                t *= m_sizes[i];

                m_total_sizes[i] = t;
            }
        }

    public:
        BINDRange &operator=(const BINDRange &) = default;

        BINDRange &operator=(BINDRange &&) = default;

        template<typename ...T>
        BINDRange(T... ts):m_sizes{ts...} {
            set_total_sizes();
        }

        BINDRangeIterator iterator(unsigned int start, unsigned int end) const {
            return BINDRangeIterator(*this, start, end);
        }

        unsigned int total_size() const {
            return m_total_sizes[D - 1];
        }

        unsigned int get_size(unsigned int v) const {
            return m_sizes[v];
        }
    };

    template<unsigned int N>
    class BINDCoordinate : public BINDRange<N> {
        using int_t = unsigned int;
        using ndrange_t = BINDRange<N>;

        std::array<int_t, N> m_positions{};
    public:
        BINDCoordinate &operator=(const BINDCoordinate &rhs) = default;

        BINDCoordinate(const BINDCoordinate &rhs) = default;

        BINDCoordinate(const std::initializer_list<std::pair<int_t, int_t>> &list) {
            std::array<int_t, N> sizes{};

            std::size_t i = 0;
            for (auto   &p: list) {
                m_positions[i] = p.first;
                sizes[i++]     = p.second;
            }

            //update the parents sizes
            static_cast<ndrange_t &>(*this) = ndrange_t(sizes);
        }

        int_t get_position(int_t d) const {
            assert(d < N);

            return m_positions[d];
        }

        void set_position(int_t d, int_t v) {
            assert(d < N);

            m_positions[d] = v;
        }

        int_t get_position_end(int_t d) const {
            return get_position(d) + ndrange_t::get_size(d);
        }
    };

    using ndrange_t = BINDRange<6>;
    using ndcoord_t = BINDCoordinate<6>;
}