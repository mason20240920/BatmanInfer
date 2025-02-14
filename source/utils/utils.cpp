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
    }
}
