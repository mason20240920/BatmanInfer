//
// Created by holynova on 2025/1/7.
//

#ifndef BATMANINFER_BI_ITERABLE_H
#define BATMANINFER_BI_ITERABLE_H

#include <iterator>

namespace BatmanInfer {

namespace utils {

namespace iterable {

    /** Reverse range iterable class
     *
     * @tparam T Type to create a reverse range on
     */
    template <typename T>
    class reverse_iterable
    {
    public:
        /** Default constructor
         *
         * @param[in] it Value to reverse iterate on
         */
        explicit reverse_iterable(T &it) : _it(it)
        {
        }

        /** Get beginning of iterator.
         *
         * @return beginning of iterator.
         */
        typename T::reverse_iterator begin()
        {
            return _it.rbegin();
        }

        /** Get end of iterator.
         *
         * @return end of iterator.
         */
        typename T::reverse_iterator end()
        {
            return _it.rend();
        }

        /** Get beginning of const iterator.
         *
         * @return beginning of const iterator.
         */
        typename T::const_reverse_iterator cbegin()
        {
            return _it.rbegin();
        }

        /** Get end of const iterator.
         *
         * @return end of const iterator.
         */
        typename T::const_reverse_iterator cend()
        {
            return _it.rend();
        }

    private:
        T &_it;
    };

    /** Creates a reverse iterable for a given type
     *
     * @tparam T Type to create a reverse iterable on
     *
     * @param[in] val Iterable input
     *
     * @return Reverse iterable container
     */
    template <typename T>
    reverse_iterable<T> reverse_iterate(T &val)
    {
        return reverse_iterable<T>(val);
    }

} // namespace iterable

} // namespace utils

} // namespace BatmanInfer

#endif //BATMANINFER_BI_ITERABLE_H
