//
// Created by holynova on 2025/1/10.
//

#ifndef BATMANINFER_GRAPH_BI_NETENSORHANDLE_H
#define BATMANINFER_GRAPH_BI_NETENSORHANDLE_H

#include "graph/bi_itensorHandle.h"
#include "runtime/bi_tensor.hpp"

namespace BatmanInfer {

namespace graph {

namespace backends {

    /** CPU Tensor handle interface object **/
    class BINETensorHandle final : public BIITensorHandle
    {
    public:
        /** Default Constructor
         *
         * @param[in] info Tensor metadata
         */
        BINETensorHandle(const BIITensorInfo &info);
        /** Destructor: free the tensor's memory */
        ~BINETensorHandle() = default;
        /** Allow instances of this class to be move constructed */
        BINETensorHandle(BINETensorHandle &&) = default;
        /** Allow instances of this class to be moved */
        BINETensorHandle &operator=(BINETensorHandle &&) = default;

        // Inherited overridden methods
        void                          allocate() override;
        void                          free() override;
        void                          manage(BIIMemoryGroup *mg) override;
        void                          map(bool blocking) override;
        void                          unmap() override;
        void                          release_if_unused() override;
        BatmanInfer::BIITensor       &tensor() override;
        const BatmanInfer::BIITensor &tensor() const override;
        BIITensorHandle              *parent_handle() override;
        bool                          is_subtensor() const override;
        BITarget                      target() const override;

    private:
        BatmanInfer::BITensor _tensor; /**< Backend Tensor */
    };

} // namespace backends

} // namespace graph

} // namespace BatmanInfer

#endif //BATMANINFER_GRAPH_BI_NETENSORHANDLE_H
