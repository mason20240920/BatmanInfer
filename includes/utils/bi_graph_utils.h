//
// Created by holynova on 25-3-16.
//

#pragma once

#include "data/core/bi_pixel_value.h"
#include "data/core/bi_utils.hpp"
#include "data/core/utils/misc/utils.hpp"
#include "graph/bi_graph.h"
#include "graph/bi_itensorAccessor.h"
#include "graph/bi_types.h"
#include "runtime/bi_tensor.hpp"

#include <array>
#include <vector>
#include <random>
#include <string>

namespace BatmanInfer {

namespace graph_utils {

    /** Numpy Binary loader class*/
    class BINumPyBinLoader final : public graph::BIITensorAccessor
    {
    public:
        /** Default Constructor
         *
         * @param[in] filename    Binary file name
         */
        explicit BINumPyBinLoader(std::string filename);
        /** Allows instances to move constructed */
        BINumPyBinLoader(BINumPyBinLoader &&) = default;

        // Inherited methods overriden:
        bool access_tensor(BIITensor &tensor) override;

    private:
        bool              _already_loaded;
        const std::string _filename;
    };

    /** Generates appropriate weights accessor according to the specified path
     * @param[in] path        Path to the data files
     * @param[in] data_file   Relative path to the data files from path
     *
     * @return An appropriate tensor accessor
     */
    inline std::unique_ptr<graph::BIITensorAccessor>
    get_weights_accessor(const std::string &path, const std::string &data_file)
    {
        return std::make_unique<NumPyBinLoader>(path + data_file);
    }

} // namespace graph_utils

} // namespace BatmanInfer