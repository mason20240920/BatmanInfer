//
// Created by holynova on 2025/1/6.
//

#ifndef BATMANINFER_GRAPH_BI_WORKLOAD_H
#define BATMANINFER_GRAPH_BI_WORKLOAD_H

#include "graph/bi_tensor.h"
#include "graph/bi_graphContext.h"
#include "runtime/bi_i_function.hpp"
#include "runtime/bi_i_memory_group.hpp"

#include <functional>
#include <memory>
#include <vector>

namespace BatmanInfer {

namespace graph {

	// Forward declarations
    class BIITensorHandle;
    class BIINode;
    class BIGraph;

    struct BIExecutionTask;

    void execute_task(BIExecutionTask &task);

    /** Task executor */
    class BITaskExecutor final
    {
    private:
        /** Default constructor **/
        BITaskExecutor();

    public:
        /** Task executor accessor
         *
         * @return Task executor instance
         */
        static BITaskExecutor &get();
        /** Function that is responsible for executing tasks */
        std::function<decltype(execute_task)> execute_function;
    };

    /** Execution task
     *
     * Contains all the information required to execute a given task
     */
    struct BIExecutionTask
    {
        BIExecutionTask(std::unique_ptr<BatmanInfer::BIIFunction> &&f, BIINode *n) : task(std::move(f)), node(n)
        {
        }
        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BIExecutionTask(const BIExecutionTask &) = delete;
        /** Prevent instances of this class from being copied (As this class contains pointers) */
        BIExecutionTask &operator=(const BIExecutionTask &) = delete;
        /** Default Move Constructor. */
        BIExecutionTask(BIExecutionTask &&) noexcept = default;
        /** Default move assignment operator */
        BIExecutionTask &operator=(BIExecutionTask &&) noexcept = default;
        /** Default destructor */
        ~BIExecutionTask() = default;
        std::unique_ptr<BatmanInfer::BIIFunction> task = {}; /**< Task to execute */
        BIINode                                  *node = {}; /**< Node bound to this workload */

        /** Function operator */
        void operator()();

        /** Prepare execution task */
        void prepare();
    };

    /** Execution workload */
    struct BIExecutionWorkload
    {
        std::vector<BITensor *>      inputs  = {};        /**< Input handles */
        std::vector<BITensor *>      outputs = {};        /**< Output handles */
        std::vector<BIExecutionTask> tasks   = {};        /**< Execution workload */
        BIGraph                     *graph   = {nullptr}; /**< Graph bound to the workload */
        BIGraphContext              *ctx     = {nullptr}; /**< Graph execution context */
    };

} // namespace graph

} // namespace BatmanInfer

#endif //BATMANINFER_GRAPH_BI_WORKLOAD_H
