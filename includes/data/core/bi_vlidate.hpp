//
// Created by Mason on 2025/1/6.
//

#ifndef BATMANINFER_BI_VALIDATE_HPP
#define BATMANINFER_BI_VALIDATE_HPP

#include <data/core/bi_error.h>
#include <data/core/bi_i_kernel.hpp>
#include <data/core/utils/data_type_utils.hpp>

#include <algorithm>

namespace BatmanInfer {

    namespace detail {
        /* Check whether two dimension objects differ.
 *
 * @param[in] dim1      First object to be compared.
 * @param[in] dim2      Second object to be compared.
 * @param[in] upper_dim The dimension from which to check.
 *
 * @return Return true if the two objects are different.
 */
        template<typename T>
        inline bool
        have_different_dimensions(const BIDimensions<T> &dim1, const BIDimensions<T> &dim2, unsigned int upper_dim) {
            for (unsigned int i = upper_dim; i < BatmanInfer::BIDimensions<T>::num_max_dimensions; ++i) {
                if (dim1[i] != dim2[i]) {
                    return true;
                }
            }

            return false;
        }

/** Function to compare two @ref Dimensions objects and throw an error on mismatch.
 *
 * @param[in] dim      Object to compare against.
 * @param[in] function Function in which the error occurred.
 * @param[in] file     File in which the error occurred.
 * @param[in] line     Line in which the error occurred.
 */
        template<typename T>
        class compare_dimension {
        public:
            /** Construct a comparison function.
             *
             * @param[in] dim      Dimensions to compare.
             * @param[in] function Source function. Used for error reporting.
             * @param[in] file     Source code file. Used for error reporting.
             * @param[in] line     Source code line. Used for error reporting.
             */
            compare_dimension(const BIDimensions<T> &dim, const char *function, const char *file, int line)
                    : _dim{dim}, _function{function}, _file{file}, _line{line} {
            }

            /** Compare the given object against the stored one.
             *
             * @param[in] dim To be compared object.
             *
             * @return a status.
             */
            BatmanInfer::BIStatus operator()(const BIDimensions<T> &dim) {
                BI_COMPUTE_RETURN_ERROR_ON_LOC_MSG(have_different_dimensions(_dim, dim, 0), _function, _file, _line,
                                                   "Objects have different dimensions");
                return BatmanInfer::BIStatus{};
            }

        private:
            const BIDimensions<T> &_dim;
            const char *const _function;
            const char *const _file;
            const int _line;
        };

        template<typename F>
        inline BatmanInfer::BIStatus for_each_error(F &&) {
            return BatmanInfer::BIStatus{};
        }

        template<typename F, typename T, typename... Ts>
        inline BatmanInfer::BIStatus for_each_error(F &&func, T &&arg, Ts &&...args) {
            BI_COMPUTE_RETURN_ON_ERROR(func(arg));
            BI_COMPUTE_RETURN_ON_ERROR(for_each_error(func, args...));
            return BatmanInfer::BIStatus{};
        }

/** Get the info for a tensor, dummy struct */
        template<typename T>
        struct get_tensor_info_t;

/** Get the info for a tensor */
        template<>
        struct get_tensor_info_t<BIITensorInfo *> {
            /** Get the info for a tensor.
             *
             * @param[in] tensor Tensor.
             *
             * @return tensor info.
             */
            BIITensorInfo *operator()(const BIITensor *tensor) {
                return tensor->info();
            }
        };


    } // namespace detail
    /**
     * @brief 如果其中一个指针是空指针，则创建一个错误。
     * @tparam Ts
     * @param function 发生错误的函数。
     * @param file Name of the file where the error occurred.
     * @param line Line on which the error occurred.
     * @param pointers Pointers to check against nullptr.
     * @return
     */
    template<typename ... Ts>
    inline BatmanInfer::BIStatus error_on_nullptr(const char *function,
                                                  const char *file,
                                                  const int line,
                                                  Ts &&...pointers) {
        const std::array<const void *, sizeof...(Ts)> pointers_array{{std::forward<Ts>(pointers)...}};
        bool has_nullptr = std::any_of(pointers_array.begin(),
                                       pointers_array.end(),
                                       [&](const void *ptr) {
                                           return (ptr == nullptr);
                                       });

        BI_COMPUTE_RETURN_ERROR_ON_LOC_MSG(has_nullptr, function, file, line, "Nullptr object!");
        return BatmanInfer::BIStatus{};
    }

#define BI_COMPUTE_ERROR_ON_NULLPTR(...) \
    BI_COMPUTE_ERROR_THROW_ON(::BatmanInfer::error_on_nullptr(__func__, __FILE__, __LINE__, __VA_ARGS__))
#define BI_COMPUTE_RETURN_ERROR_ON_NULLPTR(...) \
    BI_COMPUTE_RETURN_ON_ERROR(::BatmanInfer::error_on_nullptr(__func__, __FILE__, __LINE__, __VA_ARGS__))

    template<typename T, typename... Ts>
    BatmanInfer::BIStatus error_on_mismatching_dimensions(const char *function,
                                                          const char *file,
                                                          int line,
                                                          const BIDimensions<T> &dim1,
                                                          const BIDimensions<T> &dim2,
                                                          Ts &&...dims) {
        BI_COMPUTE_RETURN_ON_ERROR(
                detail::for_each_error(detail::compare_dimension<T>(dim1, function, file, line), dim2,
                                       std::forward<Ts>(dims)...));
        return BatmanInfer::BIStatus{};
    }

#define BI_COMPUTE_ERROR_ON_MISMATCHING_DIMENSIONS(...) \
    BI_COMPUTE_ERROR_THROW_ON(                          \
        ::BatmanInfer::error_on_mismatching_dimensions(__func__, __FILE__, __LINE__, __VA_ARGS__))
#define BI_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(...) \
    BI_COMPUTE_RETURN_ON_ERROR(                                \
        ::BatmanInfer::error_on_mismatching_dimensions(__func__, __FILE__, __LINE__, __VA_ARGS__))

    template<typename... Ts>
    inline BatmanInfer::BIStatus error_on_mismatching_quantization_info(const char *function,
                                                                        const char *file,
                                                                        const int line,
                                                                        const BIITensorInfo *tensor_info_1,
                                                                        const BIITensorInfo *tensor_info_2,
                                                                        Ts... tensor_infos) {
        BIDataType &&first_data_type = tensor_info_1->data_type();
        const BIQuantizationInfo first_quantization_info = tensor_info_1->quantization_info();

        if (!is_data_type_quantized(first_data_type)) {
            return BatmanInfer::BIStatus{};
        }

        const std::array<const BIITensorInfo *, 1 + sizeof...(Ts)> tensor_infos_array{
                {tensor_info_2, std::forward<Ts>(tensor_infos)...}};
        BI_COMPUTE_RETURN_ERROR_ON_LOC_MSG(std::any_of(tensor_infos_array.begin(), tensor_infos_array.end(),
                                                       [&](const BIITensorInfo *tensor_info) {
                                                           return tensor_info->data_type() != first_data_type;
                                                       }),
                                           function, file, line,
                                           "Tensors have different asymmetric quantized data types");
        BI_COMPUTE_RETURN_ERROR_ON_LOC_MSG(
                std::any_of(tensor_infos_array.begin(), tensor_infos_array.end(),
                            [&](const BIITensorInfo *tensor_info) {
                                return tensor_info->quantization_info() != first_quantization_info;
                            }),
                function, file, line, "Tensors have different quantization information");

        return BatmanInfer::BIStatus{};
    }

    template<typename... Ts>
    inline BatmanInfer::BIStatus error_on_mismatching_quantization_info(const char *function,
                                                                        const char *file,
                                                                        const int line,
                                                                        const BIITensor *tensor_1,
                                                                        const BIITensor *tensor_2,
                                                                        Ts... tensors) {
        BI_COMPUTE_RETURN_ON_ERROR(
                ::BatmanInfer::error_on_mismatching_quantization_info(function, file, line, tensor_1->info(),
                                                                      tensor_2->info(),
                                                                      detail::get_tensor_info_t<BIITensorInfo *>()(
                                                                              tensors)...));
        return BatmanInfer::BIStatus{};
    }

#define BI_COMPUTE_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(...) \
    BI_COMPUTE_ERROR_THROW_ON(                                 \
        ::BatmanInfer::error_on_mismatching_quantization_info(__func__, __FILE__, __LINE__, __VA_ARGS__))
#define BI_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(...) \
    BI_COMPUTE_RETURN_ON_ERROR(                                       \
        ::BatmanInfer::error_on_mismatching_quantization_info(__func__, __FILE__, __LINE__, __VA_ARGS__))

    template<typename... Ts>
    inline BatmanInfer::BIStatus error_on_mismatching_data_types(
            const char *function, const char *file, const int line, const BIITensorInfo *tensor_info,
            Ts... tensor_infos) {
        BI_COMPUTE_RETURN_ERROR_ON_LOC(tensor_info == nullptr, function, file, line);
        BI_COMPUTE_RETURN_ON_ERROR(::BatmanInfer::error_on_nullptr(function, file, line, tensor_infos...));

        BIDataType &&tensor_data_type = tensor_info->data_type();
        const std::array<const BIITensorInfo *, sizeof...(Ts)> tensors_infos_array{{tensor_infos...}};
        BI_COMPUTE_RETURN_ERROR_ON_LOC_MSG(std::any_of(tensors_infos_array.begin(), tensors_infos_array.end(),
                                                       [&](const BIITensorInfo *tensor_info_obj) {
                                                           return tensor_info_obj->data_type() != tensor_data_type;
                                                       }),
                                           function, file, line, "Tensors have different data types");
        return BatmanInfer::BIStatus{};
    }

/** Return an error if the passed two tensors have different data types
 *
 * @param[in] function Function in which the error occurred.
 * @param[in] file     Name of the file where the error occurred.
 * @param[in] line     Line on which the error occurred.
 * @param[in] tensor   The first tensor to be compared.
 * @param[in] tensors  (Optional) Further allowed tensors.
 *
 * @return Status
 */
    template<typename... Ts>
    inline BatmanInfer::BIStatus error_on_mismatching_data_types(
            const char *function, const char *file, const int line, const BIITensor *tensor, Ts... tensors) {
        BI_COMPUTE_RETURN_ERROR_ON_LOC(tensor == nullptr, function, file, line);
        BI_COMPUTE_RETURN_ON_ERROR(::BatmanInfer::error_on_nullptr(function, file, line, tensors...));
        BI_COMPUTE_RETURN_ON_ERROR(::BatmanInfer::error_on_mismatching_data_types(
                function, file, line, tensor->info(), detail::get_tensor_info_t<BIITensorInfo *>()(tensors)...));
        return BatmanInfer::BIStatus{};
    }

#define BI_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(...) \
    BI_COMPUTE_ERROR_THROW_ON(                          \
        ::BatmanInfer::error_on_mismatching_data_types(__func__, __FILE__, __LINE__, __VA_ARGS__))
#define BI_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(...) \
    BI_COMPUTE_RETURN_ON_ERROR(                                \
        ::BatmanInfer::error_on_mismatching_data_types(__func__, __FILE__, __LINE__, __VA_ARGS__))

}

#endif //BATMANINFER_BI_VALIDATE_HPP
