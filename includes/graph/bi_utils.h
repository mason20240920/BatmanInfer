//
// Created by holynova on 2025/1/13.
//

#ifndef BATMANINFER_GRAPH_BI_UTILS_H
#define BATMANINFER_GRAPH_BI_UTILS_H

#include "graph/bi_graph.h"
#include "graph/bi_passManager.h"

namespace BatmanInfer {

namespace graph {

    // Forward Declaration
    class BIGraphContext;

    inline bool is_utility_node(BIINode *node)
    {
        std::set<BINodeType> utility_node_types = {BINodeType::PrintLayer};
        return utility_node_types.find(node->type()) != utility_node_types.end();
    }

    /** Returns the tensor descriptor of a given tensor
     *
     * @param[in] g   Graph that the tensor belongs to
     * @param[in] tid Tensor ID
     *
     * @return Tensor descriptor if tensor was found else empty descriptor
     */
    inline BITensorDescriptor get_tensor_descriptor(const BIGraph &g, TensorID tid)
    {
        const BITensor *tensor = g.tensor(tid);
        return (tensor != nullptr) ? tensor->desc() : BITensorDescriptor();
    }

    /** Sets an accessor on a given tensor
     *
     * @param[in] tensor   Tensor to set the accessor to
     * @param[in] accessor Accessor to set
     *
     * @return True if accessor was set else false
     */
    inline BIStatus set_tensor_accessor(BITensor *tensor, std::unique_ptr<BIITensorAccessor> accessor)
    {
        BI_COMPUTE_RETURN_ERROR_ON(tensor == nullptr);
        tensor->set_accessor(std::move(accessor));

        return BIStatus{};
    }

    /** Checks if a specific target is supported
     *
     * @param[in] target Target to check
     *
     * @return True if target is support else false
     */
    bool is_target_supported(BITarget target);

    /** Returns default target for execution
     *
     * @note If an OpenCL backend exists then OpenCL is returned,
     *       else if the CPU backend exists returns @ref Target::NEON as target.
     *       If no backends are registered an error is raised.
     *
     * @return Default target
     */
    BITarget get_default_target();

    /** Forces a single target to all graph constructs
     *
     * @param[in] g      Graph to force target on
     * @param[in] target Target to force
     */
    void force_target_to_graph(BIGraph &g, BITarget target);

    /** Creates a default @ref PassManager
     *
     * @param[in] target Target to create the pass manager for
     * @param[in] cfg    Graph configuration meta-data
     *
     * @return A PassManager with default mutating passes
     */
    BIPassManager create_default_pass_manager(BITarget target, const BIGraphConfig &cfg);

    /** Setups requested backend context if it exists, is supported and hasn't been initialized already.
     *
     * @param[in,out] ctx    Graph Context.
     * @param[in]     target Target to setup the backend for.
     */
    void setup_requested_backend_context(BIGraphContext &ctx, BITarget target);

    /** Default releases the graph context if not done manually
     *
     * @param[in,out] ctx Graph Context
     */
    void release_default_graph_context(BIGraphContext &ctx);

    /** Synchronize kernels execution on the backends. On GPU, this results in a blocking call waiting for all kernels to be completed. */
    void sync_backends();

    /** Get size of a tensor's given dimension depending on its layout
     *
     * @param[in] descriptor            Descriptor
     * @param[in] data_layout_dimension Tensor data layout dimension
     *
     * @return Size of requested dimension
     */
    size_t get_dimension_size(const BITensorDescriptor &descriptor, const BIDataLayoutDimension data_layout_dimension);

    /** Get index of a tensor's given dimension depending on its layout
     *
     * @param[in] data_layout           Data layout of the tensor
     * @param[in] data_layout_dimension Tensor data layout dimension
     *
     * @return Idx of given dimension
     */
    size_t get_dimension_idx(BIDataLayout data_layout, const BIDataLayoutDimension data_layout_dimension);

    /** Get the list of driving nodes of a given node
     *
     * @param[in] node Node to find the driving node of
     *
     * @return A list with the driving node of a given node
     */
    std::vector<BINodeIdxPair> get_driving_nodes(const BIINode &node);

    /** Get the list of driver nodes of a given node
     *
     * @param[in] node Node to find the driver node of
     *
     * @return A list with the driver node of a given node
     */
    std::vector<BINodeIdxPair> get_driver_nodes(const BIINode &node);

    /** Configures tensor
     *
     * @param[in, out] tensor Tensor to configure
     */
    void configure_tensor(BITensor *tensor);

} // namespace graph

} // namespace BatmanInfer

#endif //BATMANINFER_GRAPH_BI_UTILS_H
