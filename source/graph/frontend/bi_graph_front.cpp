//
// Created by holynova on 2025/2/7.
//

#include "graph/frontend/bi_graph_front.h"

#include "graph/frontend/bi_ilayer.h"
#include "graph/bi_utils.h"

namespace BatmanInfer {

namespace graph {

namespace frontend {

    BIGraphFront::BIGraphFront(size_t id, std::string name) : _ctx(), _manager(), _g(id, std::move(name))
    {
    }

    void BIGraphFront::finalize(BITarget target, const BIGraphConfig &config)
    {
        BIPassManager pm = create_default_pass_manager(target, config);
        _ctx.set_config(config);
        _manager.finalize_graph(_g, _ctx, pm, target);
    }

    void BIGraphFront::run()
    {
        _manager.execute_graph(_g);
    }

    void BIGraphFront::add_layer(BIILayer &layer)
    {
        auto nid = layer.create_layer(*this);
        _tail_node = nid;
    }

    const BIGraph &BIGraphFront::graph() const
    {
        return _g;
    }

    BIGraph &BIGraphFront::graph()
    {
        return _g;
    }

} // namespace frontend

} // namespace graph

} // namespace BatmanInfer
