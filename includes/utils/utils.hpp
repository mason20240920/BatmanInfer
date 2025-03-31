//
// Created by Mason on 2025/2/14.
//

#pragma once

#include <data/core/bi_helpers.hpp>
#include <data/core/bi_i_tensor.hpp>
#include <data/core/bi_types.hpp>
#include <data/core/bi_window.hpp>
#include <runtime/bi_tensor.hpp>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wstrict-overflow"

#include <lib/npy.hpp>
#include <support/string_support.hpp>

namespace BatmanInfer {
    namespace utils {

        /** Maps a tensor if needed
        *
        * @param[in] tensor   Tensor to be mapped
        * @param[in] blocking Specified if map is blocking or not
        */
        template<typename T>
        inline void map(T &tensor, bool blocking) {
            BI_COMPUTE_UNUSED(tensor);
            BI_COMPUTE_UNUSED(blocking);
        }

        /** Unmaps a tensor if needed
        *
        * @param tensor  Tensor to be unmapped
        */
        template<typename T>
        inline void unmap(T &tensor) {
            BI_COMPUTE_UNUSED(tensor);
        }

        /**
         * Parse the npy header from an input file stream. At the end of the execution,
         * the file position pointer will be located at the first pixel stored in the npy file //TODO
         *
         * @param fs Input file stream to parse
         * @return The width and height stored in the header of the NPY file
         */
        npy::header_t parse_npy_header(std::ifstream &fs);

        /** Obtain numpy type string from DataType.
         *
         * @param[in] data_type Data type.
         *
         * @return numpy type string.
         */
        inline std::string get_typestring(BIDataType data_type) {
            // Check endianness
            const unsigned int i = 1;
            const char *c = reinterpret_cast<const char *>(&i);
            std::string endianness;
            if (*c == 1) {
                endianness = std::string("<");
            } else {
                endianness = std::string(">");
            }
            const std::string no_endianness("|");

            switch (data_type) {
                case BIDataType::QASYMM8_SIGNED:
                    return no_endianness + "i" + support::cpp11::to_string(sizeof(int8_t));
                case BIDataType::U8:
                case BIDataType::QASYMM8:
                    return no_endianness + "u" + support::cpp11::to_string(sizeof(uint8_t));
                case BIDataType::S8:
                case BIDataType::QSYMM8:
                case BIDataType::QSYMM8_PER_CHANNEL:
                    return no_endianness + "i" + support::cpp11::to_string(sizeof(int8_t));
                case BIDataType::U16:
                case BIDataType::QASYMM16:
                    return endianness + "u" + support::cpp11::to_string(sizeof(uint16_t));
                case BIDataType::S16:
                case BIDataType::QSYMM16:
                    return endianness + "i" + support::cpp11::to_string(sizeof(int16_t));
                case BIDataType::U32:
                    return endianness + "u" + support::cpp11::to_string(sizeof(uint32_t));
                case BIDataType::S32:
                    return endianness + "i" + support::cpp11::to_string(sizeof(int32_t));
                case BIDataType::U64:
                    return endianness + "u" + support::cpp11::to_string(sizeof(uint64_t));
                case BIDataType::S64:
                    return endianness + "i" + support::cpp11::to_string(sizeof(int64_t));
                case BIDataType::F16:
                    return endianness + "f" + support::cpp11::to_string(sizeof(half));
                case BIDataType::F32:
                    return endianness + "f" + support::cpp11::to_string(sizeof(float));
                case BIDataType::F64:
                    return endianness + "f" + support::cpp11::to_string(sizeof(double));
                case BIDataType::SIZET:
                    return endianness + "u" + support::cpp11::to_string(sizeof(size_t));
                default:
                    BI_COMPUTE_ERROR("Data type not supported");
            }
        }

        class NPYLoader {
        public:
            NPYLoader() : _fs(), _shape(), _fortran_order(false), _typestring() {

            }

            void open(const std::string &npy_filename) {
                BI_COMPUTE_ERROR_ON(is_open());
                try {
                    _fs.open(npy_filename, std::ios::in | std::ios::binary);
                    BI_COMPUTE_EXIT_ON_MSG_VAR(!_fs.good(), "Failed to load binary data from %s",
                                               npy_filename.c_str());
                    _fs.exceptions(std::ifstream::failbit | std::ifstream::badbit);
                    npy::header_t header = parse_npy_header(_fs);
                    _shape = header.shape;
                    _fortran_order = header.fortran_order;
                    _typestring = header.dtype.str();
                } catch (const std::ifstream::failure &e) {
                    BI_COMPUTE_ERROR_VAR("Accessing %s: %s", npy_filename.c_str(), e.what());
                }
            }

            /** Return true if a NPY file is in fortran order */
            bool is_fortran() {
                return _fortran_order;
            }

            /**
             * Return true if a NPY file is currently open
             * @return
             */
            bool is_open() {
                return _fs.is_open();
            }

            /** Initialise the tensor's metadata with the dimensions of the NPY file currently open
            *
            * @param[out] tensor Tensor to initialise
            * @param[in]  dt     Data type to use for the tensor
            */
            template<typename T>
            void init_tensor(T &tensor, BIDataType dt) {
                BI_COMPUTE_ERROR_ON(!is_open());
                BI_COMPUTE_ERROR_ON(dt != BIDataType::F32);

                // Use the size of the input NPY tensor
                BITensorShape shape;
                shape.set_num_dimensions(_shape.size());
                for (size_t i = 0; i < _shape.size(); ++i) {
                    size_t src = i;
                    if (_fortran_order) {
                        src = _shape.size() - 1 - i;
                    }
                    shape.set(i, _shape.at(src));
                }

                BITensorInfo tensor_info(shape, 1, dt);
                tensor.allocator()->init(tensor_info);
            }

            /** Fill a tensor with the content of the currently open NPY file.
             *
             * @note If the tensor is a CLTensor, the function maps and unmaps the tensor
             *
             * @param[in,out] tensor Tensor to fill (Must be allocated, and of matching dimensions with the opened NPY).
             */
            template<typename T>
            void fill_tensor(T &tensor) {
                BI_COMPUTE_ERROR_ON(!is_open());
                BI_COMPUTE_ERROR_ON_DATA_TYPE_NOT_IN(&tensor, BIDataType::QASYMM8,
                                                     BIDataType::S32,
                                                     BIDataType::F32, BIDataType::F16, BIDataType::QASYMM8_SIGNED,
                                                     BIDataType::QSYMM8_PER_CHANNEL);
                try {
                    // Map buffer if creating a CLTensor
                    map(tensor, true);

                    // Check if the file is large enough to fill the tensor
                    const size_t current_position = _fs.tellg();
                    _fs.seekg(0, std::ios_base::end);
                    const size_t end_position = _fs.tellg();
                    _fs.seekg(current_position, std::ios_base::beg);

                    BI_COMPUTE_ERROR_ON_MSG((end_position - current_position) <
                                            tensor.info()->tensor_shape().total_size() * tensor.info()->element_size(),
                                            "Not enough data in file");
                    BI_COMPUTE_UNUSED(end_position);

                    // Check if the typestring matches the given one
                    std::string expect_typestr = get_typestring(tensor.info()->data_type());

                    bool enable_f32_to_f16_conversion = false;
                    if (_typestring != expect_typestr) {
                        const std::string f32_typestring = "<f4";
                        const std::string f16_typestring = "<f2";
                        // if typestring does not match, check whether _typestring is F32 and can be downcasted to expect_typestr
                        if (_typestring == f32_typestring && expect_typestr == f16_typestring) {
                            enable_f32_to_f16_conversion = true;
                        } else {
                            BI_COMPUTE_ERROR("Typestrings mismatch");
                        }
                    }


                    // Correct dimensions (Needs to match TensorShape dimension corrections)
                    if (_shape.size() != tensor.info()->tensor_shape().num_dimensions()) {
                        for (int i = static_cast<int>(_shape.size()) - 1; i > 0; --i) {
                            if (_shape[i] == 1) {
                                _shape.pop_back();
                            } else {
                                break;
                            }
                        }
                    }

                    BITensorShape permuted_shape = tensor.info()->tensor_shape();

                    // Validate tensor shape
                    BI_COMPUTE_ERROR_ON_MSG(_shape.size() != tensor.info()->tensor_shape().num_dimensions(),
                                            "Tensor ranks mismatch");
                    for (size_t i = 0; i < _shape.size(); ++i) {
                        BI_COMPUTE_ERROR_ON_MSG(permuted_shape[i] != _shape[i], "Tensor dimensions mismatch");
                    }

                    switch (tensor.info()->data_type()) {
                        case BIDataType::QASYMM8:
                        case BIDataType::QASYMM8_SIGNED:
                        case BIDataType::QSYMM8_PER_CHANNEL:
                        case BIDataType::S32:
                        case BIDataType::F32:
                        case BIDataType::F16: {
                            // Read data
                            if (!_fortran_order && tensor.info()->padding().empty() &&
                                !enable_f32_to_f16_conversion) {
                                // If tensor has no padding read directly from stream.
                                _fs.read(reinterpret_cast<char *>(tensor.buffer()), tensor.info()->total_size());
                            } else {
                                // If tensor has padding or is in fortran order accessing tensor elements through execution window.
                                BIWindow window;
                                const unsigned int num_dims = _shape.size();
                                window.use_tensor_dimensions(permuted_shape);

                                execute_window_loop(window,
                                                    [&](const BICoordinates &id) {
                                                        BICoordinates dst(id);
                                                        if (enable_f32_to_f16_conversion) {
                                                            float f32_val = 0;
                                                            _fs.read(reinterpret_cast<char *>(&f32_val), 4u);
                                                            half f16_val =
                                                                    half_float::half_cast<half, std::round_to_nearest>(
                                                                            f32_val);
                                                            *(reinterpret_cast<half *>(tensor.ptr_to_element(
                                                                    dst))) = f16_val;
                                                        } else {
                                                            _fs.read(reinterpret_cast<char *>(tensor.ptr_to_element(
                                                                             dst)),
                                                                     tensor.info()->element_size());
                                                        }
                                                    });
                            }

                            break;
                        }
                        default:
                            BI_COMPUTE_ERROR("Unsupported data type");
                    }

                    // Unmap buffer if creating a CLTensor
                    unmap(tensor);
                }
                catch (const std::ifstream::failure &e) {
                    BI_COMPUTE_ERROR_VAR("Loading NPY file: %s", e.what());
                }
            }

        private:
            std::ifstream _fs;
            std::vector<unsigned long> _shape;
            bool _fortran_order;
            std::string _typestring;
        };


        void read_npy_to_tensor(const std::string &file_name, BITensor &tensor);

        BITensor create_tensor(const BITensorShape &shapes, BIMemoryGroup *group);

        BITensor create_npy_tensor(const std::string &file_name,
                                   const BITensorShape &shape);

        BITensor create_type_tensor(const std::string &file_name,
                                    const BITensorShape &tensor_shape,
                                    const BIDataType &type);
    }
}


