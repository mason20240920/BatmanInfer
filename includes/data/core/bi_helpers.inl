
namespace BatmanInfer {
    inline size_t get_data_layout_dimension_index(const BIDataLayoutDimension &data_layout_dimension)
    {
        const auto &dims = get_layout_map();
        const auto &it   = std::find(dims.cbegin(), dims.cend(), data_layout_dimension);
        ARM_COMPUTE_ERROR_ON_MSG(it == dims.cend(), "Invalid dimension for the given layout.");
        return it - dims.cbegin();
    }
}