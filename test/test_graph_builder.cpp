//
// Created by holynova on 25-3-16.
//

#include "glog/logging.h"
#include "gtest/gtest.h"
#include "runtime/bi_tensor.hpp"
#include "runtime/neon/bi_ne_functions.h"
#include "utils/utils.hpp"
#include "graph/bi_graph.h"
#include "graph/frontend/bi_graph_front.h"
#include "graph/frontend/bi_layers.h"
#include "support/bi_toolchain_support.hpp"
#include "utils/utils.hpp"
#include "utils/bi_graph_utils.h"

using namespace BatmanInfer;

TEST(BIBuildGraph, simple_fully_connection) {
    graph::frontend::BIGraphFront my_gf(0, "simple_fully_connection");

    my_gf.set_target(graph::BITarget::NEON);

    const BITensorShape input_shape(1, 768);
    graph::BITensorDescriptor input_descriptor = graph::BITensorDescriptor(input_shape, BIDataType::F32);

    // TODO: finish input layer
    // my_gf.add_layer_rhs(BIInputLayer);

    my_gf.add_layer_rhs(
        graph::frontend::BIFullyConnectedLayer(
            2304U,
            graph_utils::get_weights_accessor(".", "/model_files/attention_c_attn_weight.npy"),
            graph_utils::get_weights_accessor(".", "/model_files/attention_c_attn_bias.npy")
            ));

    // TODO: finish output layer
    // my_gf.add_layer_rhs(BIOutputLayer);

    // TODO: finish finalize function
    // my_gf.finalize();

    // TODO: finish run function
    // my_gf.run();

    // TODO: get model result

}


