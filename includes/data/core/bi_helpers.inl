
namespace BatmanInfer {
    inline size_t get_data_layout_dimension_index(const BIDataLayoutDimension &data_layout_dimension)
    {
        const auto &dims = get_layout_map();
        const auto &it   = std::find(dims.cbegin(), dims.cend(), data_layout_dimension);
        BI_COMPUTE_ERROR_ON_MSG(it == dims.cend(), "Invalid dimension for the given layout.");
        return it - dims.cbegin();
    }

    template <size_t dimension>
    struct IncrementIterators
    {
        template <typename T, typename ... Ts>
        static void unroll(T &&it, Ts && ...iterators) {
            auto increment = [](T &&it) {it.increment(dimension);};
            misc::utility::for_each(increment, std::forward<T>(it), std::forward<Ts>(iterators)...);
        }

        static void unroll()
        {
            // 结束迭代
        }
    };

    template <size_t dim>
    struct ForEachDimension
    {
        template <typename L, typename... Ts>
        static void unroll(const BIWindow &w, BICoordinates &id, L &&lambda_function, Ts &&...iterators)
        {
            const auto &d = w[dim - 1];

            for (auto v = d.start(); v < d.end(); v += d.step(), IncrementIterators<dim - 1>::unroll(iterators...))
            {
                id.set(dim - 1, v);
                ForEachDimension<dim - 1>::unroll(w, id, lambda_function, iterators...);
            }
        }
    };

    template <>
    struct ForEachDimension<0>
    {
        template <typename L, typename... Ts>
        static void unroll(const BIWindow &w, BICoordinates &id, L &&lambda_function, Ts &&...iterators)
        {
            BI_COMPUTE_UNUSED(w, iterators...);
            lambda_function(id);
        }
    };

    inline constexpr BIIterator::BIIterator() : _ptr(nullptr), _dims() {

    }

    inline BIIterator::BIIterator(const BIITensor *tensor, const BIWindow &window) : BIIterator() {
        BI_COMPUTE_ERROR_ON(tensor == nullptr);
        BI_COMPUTE_ERROR_ON(tensor->info() == nullptr);

        initialize(tensor->info()->num_dimensions(),
                   tensor->info()->strides_in_bytes(),
                   tensor->buffer(),
                   tensor->info()->offset_first_element_in_bytes(),
                   window);
    }

    inline BIIterator::BIIterator(size_t num_dims,
                                  const BIStrides &strides,
                                  uint8_t *buffer,
                                  size_t offset,
                                  const BIWindow &window) : BIIterator() {}

    inline void BIIterator::initialize(size_t num_dims,
                                       const BatmanInfer::BIStrides &strides,
                                       uint8_t *buffer,
                                       size_t  offset,
                                       const BatmanInfer::BIWindow &window) {
        BI_COMPUTE_ERROR_ON(buffer == nullptr);

        _ptr = buffer + offset;

        //初始化每个维度的步幅并计算迭代的第一个元素的位置：
        for (unsigned int n = 0; n < num_dims; ++n) {
            _dims[n]._stride = window[n].step() * strides[n];
            std::get<0>(_dims)._dim_start += static_cast<size_t>(strides[n]) * window[n].start();
        }

        // 拷贝开始的起点到所有维度
        for (unsigned int n = 1; n < BICoordinates::num_max_dimensions; ++n)
            _dims[n]._dim_start = std::get<0>(_dims)._dim_start;

    }

    inline void BIIterator::increment(size_t dimension) {
        BI_COMPUTE_ERROR_ON(dimension >= BICoordinates::num_max_dimensions);

        _dims[dimension]._dim_start += _dims[dimension]._stride;

        for (unsigned int n = 0; n < dimension; ++n)
            _dims[n]._dim_start = _dims[dimension]._dim_start;
    }

    inline constexpr size_t BIIterator::offset() const {
        return _dims.at(0)._dim_start;
    }

    inline constexpr uint8_t *BIIterator::ptr() const {
        return _ptr + _dims.at(0)._dim_start;
    }

    inline void BIIterator::reset(size_t dimension) {
        BI_COMPUTE_ERROR_ON(dimension >= BICoordinates::num_max_dimensions - 1);

        _dims[dimension]._dim_start = _dims[dimension + 1]._dim_start;

        for (unsigned int n = 0; n < dimension; ++n)
            _dims[n]._dim_start = _dims[dimension]._dim_start;
    }

    template <typename L, typename... Ts>
    inline void execute_window_loop(const BIWindow &w, L &&lambda_function, Ts &&...iterators)
    {
        w.validate();

        for (unsigned int i = 0; i < BICoordinates::num_max_dimensions; ++i)
        {
            BI_COMPUTE_ERROR_ON(w[i].step() == 0);
        }

        BICoordinates id;
        ForEachDimension<BICoordinates::num_max_dimensions>::unroll(w, id, std::forward<L>(lambda_function),
                                                                  std::forward<Ts>(iterators)...);
    }
}