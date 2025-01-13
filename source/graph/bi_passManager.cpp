//
// Created by holynova on 2025/1/13.
//

#include "graph/bi_passManager.h"

#include "graph/bi_logger.h"

namespace BatmanInfer {

namespace graph {

    BIPassManager::BIPassManager() : _passes()
    {
    }

    const std::vector<std::unique_ptr<BIIGraphMutator>> &BIPassManager::passes() const
    {
        return _passes;
    }

    BIIGraphMutator *BIPassManager::pass(size_t index)
    {
        return (index >= _passes.size()) ? nullptr : _passes.at(index).get();
    }

    void BIPassManager::append(std::unique_ptr<BIIGraphMutator> pass, bool conditional)
    {
        if (pass && conditional)
        {
            BI_COMPUTE_LOG_GRAPH_VERBOSE("Appending mutating pass : " << pass->name() << std::endl);
            _passes.push_back(std::move(pass));
        }
    }

    void BIPassManager::clear()
    {
        _passes.clear();
    }

    void BIPassManager::run_all(BIGraph &g)
    {
        for (auto &pass : _passes)
        {
            if (pass)
            {
                BI_COMPUTE_LOG_GRAPH_INFO("Running mutating pass : " << pass->name() << std::endl);
                pass->mutate(g);
            }
        }
    }

    void BIPassManager::run_type(BIGraph &g, BIIGraphMutator::MutationType type)
    {
        for (auto &pass : _passes)
        {
            if (pass && (pass->type() == type))
            {
                BI_COMPUTE_LOG_GRAPH_INFO("Running mutating pass : " << pass->name() << std::endl);
                pass->mutate(g);
            }
        }
    }

    void BIPassManager::run_index(BIGraph &g, size_t index)
    {
        if (index >= _passes.size())
        {
            return;
        }

        auto &pass = _passes.at(index);
        if (pass != nullptr)
        {
            BI_COMPUTE_LOG_GRAPH_INFO("Running mutating pass : " << pass->name() << std::endl);
            pass->mutate(g);
        }
    }

} // namespace graph

} // namespace BatmanInfer
