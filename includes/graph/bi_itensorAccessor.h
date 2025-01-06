//
// Created by holynova on 2025/1/6.
//

#ifndef BATMANINFER_GRAPH_BI_ITENSORACCESSOR_H
#define BATMANINFER_GRAPH_BI_ITENSORACCESSOR_H

#include "data/core/bi_i_tensor.hpp"

#include <memory>

namespace BatmanInfer {

namespace graph {

    /** Tensor accessor interface */
    class BIITensorAccessor
    {
    public:
        /** Default virtual destructor */
        virtual ~BIITensorAccessor() = default;
        /** Interface to be implemented to access a given tensor
         *
         * @param[in] tensor Tensor to be accessed
         *
         * @return True if access is successful else false
         */
        virtual bool access_tensor(BIITensor &tensor) = 0;
        /** Returns true if the tensor data is being accessed
         *
         * @return True if the tensor data is being accessed by the accessor. False otherwise
         */
        virtual bool access_tensor_data()
        {
            return true;
        }
    };

    using BIITensorAccessorUPtr = std::unique_ptr<BIITensorAccessor>;

} // namespace graph

} // namespace BatmanInfer

#endif //BATMANINFER_GRAPH_BI_ITENSORACCESSOR_H
