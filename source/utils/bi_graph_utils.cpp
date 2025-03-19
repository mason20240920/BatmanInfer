//
// Created by holynova on 25-3-16.
//

#include "utils/bi_graph_utils.h"

#include "data/core/bi_helpers.hpp"
#include "data/core/bi_types.hpp"
#include "graph/bi_logger.h"
#include "utils/utils.hpp"

#include <inttypes.h>
#include <iomanip>
#include <limits>

using namespace BatmanInfer::graph_utils;


BINumPyBinLoader::BINumPyBinLoader(std::string filename)
    : _already_loaded(false), _filename(std::move(filename))
{
}

bool BINumPyBinLoader::access_tensor(BIITensor &tensor)
{
    if (!_already_loaded)
    {
        utils::NPYLoader loader;
        loader.open(_filename);
        loader.fill_tensor(tensor);
    }

    _already_loaded = !_already_loaded;
    return _already_loaded;
}
