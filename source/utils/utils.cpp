//
// Created by Mason on 2025/2/14.
//

#include <utils/utils.hpp>

namespace BatmanInfer {
    namespace utils {
        npy::header_t parse_npy_header(std::ifstream &fs) {
            // Read header
            std::string header_s = npy::read_header(fs);

            // Parse header
            npy::header_t header = npy::parse_header(header_s);

            bool fortran_order = false;
            std::vector<unsigned long> shape = header.shape;

            std::reverse(shape.begin(), shape.end());

            return npy::header_t{header.dtype, fortran_order, shape};
        }


        void read_npy_to_tensor(const std::string &file_name, BITensor &tensor) {
            NPYLoader loader;
            loader.open(file_name);
            loader.fill_tensor(tensor);
        }

        BITensor create_tensor(const BITensorShape &shapes, BIMemoryGroup *group) {
            const BITensorInfo input_info(shapes, 1, BIDataType::F16);
            BITensor input;
            input.allocator()->init(input_info);
            if (group != nullptr)
                group->manage(&input);
            input.allocator()->allocate();
            return input;
        }

        BITensor create_npy_tensor(const std::string &file_name,
                                   const BITensorShape &shape) {
            BITensor tensor;
            BITensorInfo tensor_info(shape, 1, BIDataType::F16);
            tensor.allocator()->init(tensor_info);
            tensor.allocator()->allocate();
            utils::read_npy_to_tensor(file_name, tensor);

            return tensor;
        }

        BITensor create_type_tensor(const std::string &file_name,
                                    const BITensorShape &tensor_shape,
                                    const BIDataType &type) {
            BITensor tensor;
            BITensorInfo tensor_info(tensor_shape, 1, type);
            tensor.allocator()->init(tensor_info);
            tensor.allocator()->allocate();
            utils::read_npy_to_tensor(file_name, tensor);

            return tensor;
        }
    }
}
