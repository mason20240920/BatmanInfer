//
// Created by holynova on 25-4-19.
//

#include "model_interface/gpt2_model.h"

#include "kv_cache_manager/bi_kv_cache_manager.hpp"
#include "runtime/neon/bi_ne_scheduler.hpp"

// ======================================== for debug part ========================================

#define TO_STR(value) #value

void debug_print_tensor(const BITensor &tensor, std::string name) {
    std::cout << name << ":" << std::endl;
    BatmanInfer::BIIOFormatInfo format;
    format.element_delim = ", "; // 元素之间用逗号分隔
    format.row_delim = "\n"; // 每行换行
    format.align_columns = true; // 对齐列
    format.print_region = BatmanInfer::BIIOFormatInfo::PrintRegion::Full;

    tensor.print(std::cout, format);
}

// ======================================== public function part ========================================

BIGPT2Model::BIGPT2Model(std::shared_ptr<BIIMemoryManager> memory_manager) : _memory_group(std::move(memory_manager)) {
    // _output_positions.clear();
}

BIGPT2Model::BIGPT2Model() : BIGPT2Model(BIMemoryManagerOnDemand::make_default()) {
}

BIErrCode BIGPT2Model::bi_init(const char *data_in, size_t data_size) {
    // check input parameters
    if (!data_in || !data_size) {
        return BIErrCode::BIInvalidArgument;
    }

    OrderPtrMap order2ptr;
    // parse model data
    auto ret = parse_model_data(data_in, data_size, order2ptr);
    CHECK_SUCCESS(ret);

    // load all weight tensors
    ret = load_all_non_dynamic_tensors(order2ptr);
    CHECK_SUCCESS(ret);

    // initialize all intermediate tensors
    const std::vector<int> init_tensor_shape = {1, 1};
    ret = set_all_intermediate_tensors(init_tensor_shape);
    CHECK_SUCCESS(ret);

    // configure all layers initially
    ret = init_configure_all_layers(init_tensor_shape);
    CHECK_SUCCESS(ret);

    // 为了使内存正确分配，需要在开始的时候 run 一次，以执行每一个算子中的 prepare 函数
    std::vector<std::vector<unsigned int> > tmp_input_vec = {{0}};
    fill_tensor_data_2D(_sub_input_tensor, tmp_input_vec, 1);

    std::vector< std::vector<float> > output_vec;
    ret = bi_run(output_vec);
    CHECK_SUCCESS(ret);

    return ret;
}

BIErrCode BIGPT2Model::bi_set_input(std::vector<std::vector<unsigned int> > &input_vec) {
    auto ret = BIErrCode::BISuccess;

    int cur_batch_size = static_cast<int>(input_vec.size()), cur_seq_len = 0;
    for (auto &item: input_vec) {
        int item_len = static_cast<int>(item.size());
        cur_seq_len = cur_seq_len > item_len ? cur_seq_len : item_len;
    }

    const std::vector<int> tensor_shape = {cur_seq_len, cur_batch_size};

    // adjust tensor shapes
    ret = set_all_intermediate_tensors(tensor_shape);
    CHECK_SUCCESS(ret);

    // set input value
    ret = fill_tensor_data_2D(_sub_input_tensor, input_vec, cur_seq_len);
    CHECK_SUCCESS(ret);

    // configure some layers dynamically
    ret = dynamic_configure_all_layers(tensor_shape);
    CHECK_SUCCESS(ret);

    return ret;
}

BIErrCode BIGPT2Model::bi_run(std::vector<std::vector<float> > &output_vec) {
    auto ret = BIErrCode::BISuccess;

    try {
        _gather_layer.run();
        print_tensor(_sub_gather_output_tensor, "_sub_gather_output_tensor");

        _add_layer.run();
        print_tensor(_sub_add_output_tensor, "_sub_add_output_tensor");

        _gpt_multi_block_layer.run();
        print_tensor(_sub_multi_gpt_o_tensor, "_sub_multi_gpt_o_tensor");

        _final_layernorm_layer.run();
        print_tensor(_sub_final_ln_o_tensor, "_sub_final_ln_o_tensor");

        _lm_head_layer.run();
        print_tensor(_sub_lm_head_output_tensor, "_sub_lm_head_output_tensor");
    } catch (std::runtime_error &e) {
        ret = BIErrCode::BIGeneralError;
        std::cout << std::string(__FUNCTION__) << " ERROR: " << e.what() << std::endl;
    }
    CHECK_SUCCESS(ret);

    // fill output_vec
    output_vec.clear();
    const int cur_seq_len = static_cast<int>(_sub_lm_head_output_tensor.info()->tensor_shape()[BIWindow::DimY]);
    const size_t cur_batch_size = _sub_lm_head_output_tensor.info()->tensor_shape()[BIWindow::DimZ];

    output_vec.resize(cur_batch_size);
    const auto result_ptr = reinterpret_cast<half *>(_sub_lm_head_output_tensor.buffer());

    for (size_t item = 0; item < cur_batch_size; ++item) {
        output_vec[item].reserve(class_num);

        const auto cur_result_ptr = result_ptr + ((item + 1) * cur_seq_len - 1) * class_num;

        for (size_t prob_idx = 0; prob_idx < class_num; ++prob_idx) {
            output_vec[item].push_back(static_cast<float>(cur_result_ptr[prob_idx]));
        }
    }

    return ret;
}

void BIGPT2Model::set_threads_num(unsigned int num_threads) {
    BIScheduler::get().set_num_threads(num_threads);
};

// ======================================== private function part ========================================

template<typename T>
BIErrCode BIGPT2Model::fill_tensor_data_1D(BITensor &tensor, std::vector<T> &data_in) {
    const size_t tensor_element_size = tensor.info()->tensor_shape().total_size();
    const size_t input_element_size = data_in.size();

    if (tensor_element_size != input_element_size) {
        return BIErrCode::BISizeNotMatch;
    }

    auto tensor_ptr = reinterpret_cast<T *>(tensor.buffer());

    for (auto i = 0; i < tensor_element_size; ++i) {
        tensor_ptr[i] = data_in[i];
    }

    return BIErrCode::BISuccess;
}

template<typename T>
BIErrCode BIGPT2Model::fill_tensor_data_2D(BITensor &tensor, std::vector<std::vector<T> > &data_in) {
    size_t max_item_len = 0;
    for (auto &item: data_in) {
        max_item_len = item.size() > max_item_len ? item.size() : max_item_len;
    }

    fill_tensor_data_2D(tensor, data_in, max_item_len);

    return BIErrCode::BISuccess;
}

template<typename T>
BIErrCode BIGPT2Model::fill_tensor_data_2D(BITensor &tensor, std::vector<std::vector<T> > &data_in,
                                           const size_t max_item_len) {
    const size_t input_item_cnt = data_in.size();

    if ((tensor.info()->tensor_shape()[0] != max_item_len) ||
        (tensor.info()->tensor_shape()[1] != input_item_cnt)) {
        return BIErrCode::BISizeNotMatch;
    }

    auto tensor_ptr = reinterpret_cast<T *>(tensor.buffer());
    for (auto i = 0; i < input_item_cnt; ++i) {
        const size_t item_len = data_in[i].size();
        for (auto j = 0; j < max_item_len; ++j) {
            if (j < item_len) {
                tensor_ptr[i * max_item_len + j] = data_in[i][j];
            } else {
                tensor_ptr[i * max_item_len + j] = 2;
            }
        }
    }

    return BIErrCode::BISuccess;
}

BIErrCode BIGPT2Model::parse_model_data(const char *data_in, size_t data_size, OrderPtrMap &order2ptr) {
    order2ptr.clear();

    char *tmp_ptr = const_cast<char *>(data_in);
    int tmp_size = static_cast<int>(data_size);

    // 循环查找每一个资源的起始地址
    while (tmp_size > 0) {
        auto header = reinterpret_cast<GPT2ResHeader *>(tmp_ptr);
        if (header->res_order > static_cast<unsigned int>(GPT2ResOrder::all_res_count)) {
            return BIErrCode::BIResDamaged;
        }

        order2ptr[static_cast<GPT2ResOrder>(header->res_order)] = tmp_ptr;

        tmp_ptr += (sizeof(GPT2ResHeader) + header->data_length);
        tmp_size -= (static_cast<int>(sizeof(GPT2ResHeader)) + static_cast<int>(header->data_length));
    }

    return BIErrCode::BISuccess;
}

BIErrCode BIGPT2Model::load_weight_tensor(BITensor &tensor, GPT2ResOrder res_order, OrderPtrMap &order2ptr) {
    if (order2ptr.find(res_order) == order2ptr.end()) {
        return BIErrCode::BIResNotExists;
    }

    char *tmp_ptr = order2ptr[res_order];

    auto header = reinterpret_cast<GPT2ResHeader *>(tmp_ptr);

    BITensor tensor_tmp;

    // 检查 shape 是否对得上
    for (size_t i = 0; i < tensor.info()->num_dimensions(); ++i) {
        if (tensor.info()->tensor_shape()[i] != header->shape[i]) {
            return BIErrCode::BIResDamaged;
        }
    }
    for (auto i = tensor.info()->num_dimensions(); i < tensor_max_dim; ++i) {
        if (header->shape[i] != 1) {
            return BIErrCode::BIResDamaged;
        }
    }

    // shape 检查完，其实这一步可以不用再检查了，但是保险起见还是再检查一遍
    if (tensor.info()->total_size() != header->data_length) {
        return BIErrCode::BIResDamaged;
    }

    // 检查类型是否对得上
    std::string tensor_type_str = BatmanInfer::utils::get_typestring(tensor.info()->data_type());
    if (tensor_type_str != std::string(header->data_type)) {
        return BIErrCode::BIResDamaged;
    }

    // 拷贝具体数据
    tmp_ptr += sizeof(GPT2ResHeader);
    memcpy(reinterpret_cast<char *>(tensor.buffer()), tmp_ptr, header->data_length);

    return BIErrCode::BISuccess;
}

BIErrCode BIGPT2Model::load_weight_tensors(std::array<BITensor, layer_num> &tensors, GPT2ResOrder res_order, OrderPtrMap &order2ptr, int step) {
    // 多层模型，中间重复类型权重需要循环进行加载
    for (int i = 0; i < layer_num; ++i) {
        if (order2ptr.find(static_cast<GPT2ResOrder>(static_cast<int>(res_order) + i*step)) == order2ptr.end()) {
            return BIErrCode::BIResNotExists;
        }

        char *tmp_ptr = order2ptr[static_cast<GPT2ResOrder>(static_cast<int>(res_order) + i*step)];

        auto header = reinterpret_cast<GPT2ResHeader *>(tmp_ptr);

        BITensor tensor_tmp;
        // 检查 shape 是否对得上
        for (size_t j = 0; j < tensors[i].info()->num_dimensions(); ++j) {
            if (tensors[i].info()->tensor_shape()[j] != header->shape[j]) {
                return BIErrCode::BIResDamaged;
            }
        }
        for (auto j = tensors[i].info()->num_dimensions(); j < tensor_max_dim; ++j) {
            if (header->shape[j] != 1) {
                return BIErrCode::BIResDamaged;
            }
        }

        // shape 检查完，其实这一步可以不用再检查了，但是保险起见还是再检查一遍
        if (tensors[i].info()->total_size() != header->data_length) {
            return BIErrCode::BIResDamaged;
        }

        // 检查类型是否对得上
        std::string tensor_type_str = BatmanInfer::utils::get_typestring(tensors[i].info()->data_type());
        if (tensor_type_str != std::string(header->data_type)) {
            return BIErrCode::BIResDamaged;
        }

        // 拷贝具体数据
        tmp_ptr += sizeof(GPT2ResHeader);
        memcpy(reinterpret_cast<char *>(tensors[i].buffer()), tmp_ptr, header->data_length);
    }

    return BIErrCode::BISuccess;
}


BIErrCode BIGPT2Model::load_all_non_dynamic_tensors(OrderPtrMap &order2ptr) {
    auto ret = BIErrCode::BISuccess;

    const BITensorShape ori_input_tensor_shape(max_seq_len, max_batch_size);
    _ori_input_tensor.allocator()->init(BITensorInfo(ori_input_tensor_shape, 1, BIDataType::U32));

    const BITensorShape gather_weight_tensor_shape(hidden_size, dict_size);
    _gather_weight_tensor.allocator()->init(BITensorInfo(gather_weight_tensor_shape, 1, BIDataType::F16));

    const BITensorShape ori_gather_output_tensor_shape(hidden_size, max_seq_len, max_batch_size);
    _ori_gather_output_tensor.allocator()->init(BITensorInfo(ori_gather_output_tensor_shape, 1, BIDataType::F16));

    const BITensorShape add_weight_tensor_shape(hidden_size, max_seq_len);
    _add_weight_tensor.allocator()->init(BITensorInfo(add_weight_tensor_shape, 1, BIDataType::F16));

    _ori_add_output_tensor.allocator()->init(BITensorInfo(ori_gather_output_tensor_shape, 1, BIDataType::F16));

    const BITensorShape attn_gamma_weight_tensor_shape(hidden_size);
    for (int i = 0; i < layer_num; ++i) {
        BITensor attn_gamma_weight_tensor_tmp;
        _attn_gamma_weight_tensors[i].allocator()->init(BITensorInfo(attn_gamma_weight_tensor_shape, 1, BIDataType::F16));
    }
    for (int i = 0; i < layer_num; ++i) {
        BITensor attn_gamma_bias_tensor_tmp;
        _attn_gamma_bias_tensors[i].allocator()->init(BITensorInfo(attn_gamma_weight_tensor_shape, 1, BIDataType::F16));
    }

    const BITensorShape c_attn_weight_tensor_shape(hidden_size * 3, hidden_size);
    for (int i = 0; i < layer_num; ++i) {
        BITensor c_attn_weight_tensor_tmp;
        _c_attn_weight_tensors[i].allocator()->init(BITensorInfo(c_attn_weight_tensor_shape, 1, BIDataType::F16));
    }

    const BITensorShape c_attn_bias_tensor_shape(hidden_size * 3);
    for (int i = 0; i < layer_num; ++i) {
        BITensor c_attn_bias_tensor_tmp;
        _c_attn_bias_tensors[i].allocator()->init(BITensorInfo(c_attn_bias_tensor_shape, 1, BIDataType::F16));
    }

    const BITensorShape p_attn_weight_tensor_shape(hidden_size, hidden_size);
    for (int i = 0; i < layer_num; ++i) {
        BITensor p_attn_weight_tensor_tmp;
        _p_attn_weight_tensors[i].allocator()->init(BITensorInfo(p_attn_weight_tensor_shape, 1, BIDataType::F16));
    }

    const BITensorShape p_attn_bias_tensor_shape(hidden_size);
    for (int i = 0; i < layer_num; ++i) {
        BITensor p_attn_bias_tensor_tmp;
        _p_attn_bias_tensors[i].allocator()->init(BITensorInfo(p_attn_bias_tensor_shape, 1, BIDataType::F16));
    }
    for (int i = 0; i < layer_num; ++i) {
        BITensor mlp_weight_tensor_tmp;
        _mlp_weight_tensors[i].allocator()->init(BITensorInfo(p_attn_bias_tensor_shape, 1, BIDataType::F16));
    }
    for (int i = 0; i < layer_num; ++i) {
        BITensor mlp_bias_tensor_tmp;
        _mlp_bias_tensors[i].allocator()->init(BITensorInfo(p_attn_bias_tensor_shape, 1, BIDataType::F16));
    }

    const BITensorShape c_fc_weight_tensor_shape(hidden_size * 4, hidden_size);
    for (int i = 0; i < layer_num; ++i) {
        BITensor c_fc_weight_tensor_tmp;
        _c_fc_weight_tensors[i].allocator()->init(BITensorInfo(c_fc_weight_tensor_shape, 1, BIDataType::F16));
    }

    const BITensorShape c_fc_bias_tensor_shape(hidden_size * 4);
    for (int i = 0; i < layer_num; ++i) {
        BITensor c_fc_bias_tensor_tmp;
        _c_fc_bias_tensors[i].allocator()->init(BITensorInfo(c_fc_bias_tensor_shape, 1, BIDataType::F16));
    }

    const BITensorShape c_proj_weight_tensor_shape(hidden_size, hidden_size * 4);
    for (int i = 0; i < layer_num; ++i) {
        BITensor c_proj_weight_tensor_tmp;
        _c_proj_weight_tensors[i].allocator()->init(BITensorInfo(c_proj_weight_tensor_shape, 1, BIDataType::F16));
    }

    const BITensorShape c_proj_bias_tensor_shape(hidden_size);
    for (int i = 0; i < layer_num; ++i) {
        BITensor c_proj_bias_tensor_tmp;
        _c_proj_bias_tensors[i].allocator()->init(BITensorInfo(c_proj_bias_tensor_shape, 1, BIDataType::F16));
    }

    const BITensorShape gpt_o_tensor_shape(hidden_size, max_seq_len, max_batch_size);
    _ori_multi_gpt_o_tensor.allocator()->init(BITensorInfo(gpt_o_tensor_shape, 1, BIDataType::F16));

    const BITensorShape final_layernorm_tensor_shape(hidden_size);
    _final_layernorm_weight_tensor.allocator()->init(BITensorInfo(final_layernorm_tensor_shape, 1, BIDataType::F16));
    _final_layernorm_bias_tensor.allocator()->init(BITensorInfo(final_layernorm_tensor_shape, 1, BIDataType::F16));

    _ori_final_ln_o_tensor.allocator()->init(BITensorInfo(gpt_o_tensor_shape, 1, BIDataType::F16));

    const BITensorShape lm_score_weight_tensor_shape(class_num, hidden_size);
    _lm_score_weight_tensor.allocator()->init(BITensorInfo(lm_score_weight_tensor_shape, 1, BIDataType::F16));

    const BITensorShape lm_head_output_tensor_shape(class_num, max_seq_len, max_batch_size);
    _ori_lm_head_output_tensor.allocator()->init(BITensorInfo(lm_head_output_tensor_shape, 1, BIDataType::F16));

    // 内存统一管理
    _memory_group.manage(&_ori_input_tensor);
    _memory_group.manage(&_gather_weight_tensor);
    _memory_group.manage(&_ori_gather_output_tensor);
    _memory_group.manage(&_add_weight_tensor);
    _memory_group.manage(&_ori_add_output_tensor);
    for (int i = 0; i < _attn_gamma_weight_tensors.size(); ++i) {
        _memory_group.manage(&(_attn_gamma_weight_tensors[i]));
    }
    for (int i = 0; i < _attn_gamma_bias_tensors.size(); ++i) {
        _memory_group.manage(&(_attn_gamma_bias_tensors[i]));
    }
    for (int i = 0; i < _c_attn_weight_tensors.size(); ++i) {
        _memory_group.manage(&(_c_attn_weight_tensors[i]));
    }
    for (int i = 0; i < _c_attn_bias_tensors.size(); ++i) {
        _memory_group.manage(&(_c_attn_bias_tensors[i]));
    }
    for (int i = 0; i < _p_attn_weight_tensors.size(); ++i) {
        _memory_group.manage(&(_p_attn_weight_tensors[i]));
    }
    for (int i = 0; i < _p_attn_bias_tensors.size(); ++i) {
        _memory_group.manage(&(_p_attn_bias_tensors[i]));
    }
    for (int i = 0; i < _mlp_weight_tensors.size(); ++i) {
        _memory_group.manage(&(_mlp_weight_tensors[i]));
    }
    for (int i = 0; i < _mlp_bias_tensors.size(); ++i) {
        _memory_group.manage(&(_mlp_bias_tensors[i]));
    }
    for (int i = 0; i < _c_fc_weight_tensors.size(); ++i) {
        _memory_group.manage(&(_c_fc_weight_tensors[i]));
    }
    for (int i = 0; i < _c_fc_bias_tensors.size(); ++i) {
        _memory_group.manage(&(_c_fc_bias_tensors[i]));
    }
    for (int i = 0; i < _c_proj_weight_tensors.size(); ++i) {
        _memory_group.manage(&(_c_proj_weight_tensors[i]));
    }
    for (int i = 0; i < _c_proj_bias_tensors.size(); ++i) {
        _memory_group.manage(&(_c_proj_bias_tensors[i]));
    }
    _memory_group.manage(&_ori_multi_gpt_o_tensor);
    _memory_group.manage(&_final_layernorm_weight_tensor);
    _memory_group.manage(&_final_layernorm_bias_tensor);
    _memory_group.manage(&_ori_final_ln_o_tensor);
    _memory_group.manage(&_lm_score_weight_tensor);
    _memory_group.manage(&_ori_lm_head_output_tensor);


    _ori_input_tensor.allocator()->allocate();
    _gather_weight_tensor.allocator()->allocate();
    _ori_gather_output_tensor.allocator()->allocate();
    _add_weight_tensor.allocator()->allocate();
    _ori_add_output_tensor.allocator()->allocate();
    for (int i = 0; i < _attn_gamma_weight_tensors.size(); ++i) {
        _attn_gamma_weight_tensors[i].allocator()->allocate();
    }
    for (int i = 0; i < _attn_gamma_bias_tensors.size(); ++i) {
        _attn_gamma_bias_tensors[i].allocator()->allocate();
    }
    for (int i = 0; i < _c_attn_weight_tensors.size(); ++i) {
        _c_attn_weight_tensors[i].allocator()->allocate();
    }
    for (int i = 0; i < _c_attn_bias_tensors.size(); ++i) {
        _c_attn_bias_tensors[i].allocator()->allocate();
    }
    for (int i = 0; i < _p_attn_weight_tensors.size(); ++i) {
        _p_attn_weight_tensors[i].allocator()->allocate();
    }
    for (int i = 0; i < _p_attn_bias_tensors.size(); ++i) {
        _p_attn_bias_tensors[i].allocator()->allocate();
    }
    for (int i = 0; i < _mlp_weight_tensors.size(); ++i) {
        _mlp_weight_tensors[i].allocator()->allocate();
    }
    for (int i = 0; i < _mlp_bias_tensors.size(); ++i) {
        _mlp_bias_tensors[i].allocator()->allocate();
    }
    for (int i = 0; i < _c_fc_weight_tensors.size(); ++i) {
        _c_fc_weight_tensors[i].allocator()->allocate();
    }
    for (int i = 0; i < _c_fc_bias_tensors.size(); ++i) {
        _c_fc_bias_tensors[i].allocator()->allocate();
    }
    for (int i = 0; i < _c_proj_weight_tensors.size(); ++i) {
        _c_proj_weight_tensors[i].allocator()->allocate();
    }
    for (int i = 0; i < _c_proj_bias_tensors.size(); ++i) {
        _c_proj_bias_tensors[i].allocator()->allocate();
    }
    _ori_multi_gpt_o_tensor.allocator()->allocate();
    _final_layernorm_weight_tensor.allocator()->allocate();
    _final_layernorm_bias_tensor.allocator()->allocate();
    _ori_final_ln_o_tensor.allocator()->allocate();
    _lm_score_weight_tensor.allocator()->allocate();
    _ori_lm_head_output_tensor.allocator()->allocate();

    _scope_manager = std::make_unique<BIMemoryGroupResourceScope>(_memory_group);

    // load gather weight
    ret = load_weight_tensor(_gather_weight_tensor, GPT2ResOrder::transformer_wte_weight, order2ptr);
    CHECK_SUCCESS(ret);

    // load add weight
    ret = load_weight_tensor(_add_weight_tensor, GPT2ResOrder::add_wte_weight, order2ptr);
    CHECK_SUCCESS(ret);

    // load gamma weights
    ret = load_weight_tensors(_attn_gamma_weight_tensors, GPT2ResOrder::attn_layernorm_weight_0, order2ptr, 12);
    CHECK_SUCCESS(ret);

    // load gamma bias
    ret = load_weight_tensors(_attn_gamma_bias_tensors, GPT2ResOrder::attn_layernorm_bias_0, order2ptr, 12);
    CHECK_SUCCESS(ret);

    // load c attn weights
    ret = load_weight_tensors(_c_attn_weight_tensors, GPT2ResOrder::c_attn_weights_0, order2ptr, 12);
    CHECK_SUCCESS(ret);

    // load c attn bias
    ret = load_weight_tensors(_c_attn_bias_tensors, GPT2ResOrder::c_attn_bias_0, order2ptr, 12);
    CHECK_SUCCESS(ret);

    // load p attn weights
    ret = load_weight_tensors(_p_attn_weight_tensors, GPT2ResOrder::p_attn_weights_0, order2ptr, 12);
    CHECK_SUCCESS(ret);

    // load p attn bias
    ret = load_weight_tensors(_p_attn_bias_tensors, GPT2ResOrder::p_attn_bias_0, order2ptr, 12);
    CHECK_SUCCESS(ret);

    // load mlp weights
    ret = load_weight_tensors(_mlp_weight_tensors, GPT2ResOrder::mlp_layernorm_bias_0, order2ptr, 12);
    CHECK_SUCCESS(ret);

    // load mlp bias
    ret = load_weight_tensors(_mlp_bias_tensors, GPT2ResOrder::mlp_layernorm_bias_0, order2ptr, 12);
    CHECK_SUCCESS(ret);

    // load c fc weight
    ret = load_weight_tensors(_c_fc_weight_tensors, GPT2ResOrder::reordered_c_fc_weights_0, order2ptr, 12);
    CHECK_SUCCESS(ret);

    // load c fc bias
    ret = load_weight_tensors(_c_fc_bias_tensors, GPT2ResOrder::c_fc_bias_0, order2ptr, 12);
    CHECK_SUCCESS(ret);

    // load c proj weight
    ret = load_weight_tensors(_c_proj_weight_tensors, GPT2ResOrder::c_proj_weights_0, order2ptr, 12);
    CHECK_SUCCESS(ret);

    // load c proj bias
    ret = load_weight_tensors(_c_proj_bias_tensors, GPT2ResOrder::c_proj_bias_0, order2ptr, 12);
    CHECK_SUCCESS(ret);

    // load final layernorm weight
    ret = load_weight_tensor(_final_layernorm_weight_tensor, GPT2ResOrder::final_layernorm_weights, order2ptr);
    CHECK_SUCCESS(ret);

    // load final layernorm bias
    ret = load_weight_tensor(_final_layernorm_bias_tensor, GPT2ResOrder::final_layernorm_bias, order2ptr);
    CHECK_SUCCESS(ret);

    // load lm score weight
    ret = load_weight_tensor(_lm_score_weight_tensor, GPT2ResOrder::lm_score_weights, order2ptr);
    CHECK_SUCCESS(ret);

    return BIErrCode::BISuccess;
}


BIErrCode BIGPT2Model::set_all_intermediate_tensors(const std::vector<int> &tensor_shape) {
    auto ret = BIErrCode::BISuccess;

    int cur_batch_size = 1, cur_seq_len = 1;
    if (tensor_shape.size() >= 1) {
        cur_seq_len = tensor_shape[0];
    }
    if (tensor_shape.size() >= 2) {
        cur_batch_size = tensor_shape[1];
    }

    if ((cur_batch_size > max_batch_size) || (cur_seq_len > max_seq_len)) {
        return BIErrCode::BISizeNotMatch;
    }

    BITensorShape sub_input_tensor_shape(cur_seq_len, cur_batch_size);
    BITensorInfo sub_input_tensor_info(sub_input_tensor_shape, 1, BIDataType::U32);
    sub_input_tensor_info.set_format(Format::U32);
    _sub_input_tensor.allocator()->init(*_ori_input_tensor.allocator(), sub_input_tensor_info);

    BITensorShape sub_gather_output_tensor_shape(hidden_size, cur_seq_len, cur_batch_size);
    BITensorInfo sub_gather_output_tensor_info(sub_gather_output_tensor_shape, 1, BIDataType::F16);
    sub_gather_output_tensor_info.set_format(Format::F16);
    _sub_gather_output_tensor.allocator()->init(*_ori_gather_output_tensor.allocator(), sub_gather_output_tensor_info);

    BITensorShape sub_add_weight_tensor_shape(hidden_size, cur_seq_len);
    BITensorInfo sub_add_weight_tensor_info(sub_add_weight_tensor_shape, 1, BIDataType::F16);
    sub_add_weight_tensor_info.set_format(Format::F16);
    _sub_add_weight_tensor.allocator()->init(*_add_weight_tensor.allocator(), sub_add_weight_tensor_info);

    _sub_add_output_tensor.allocator()->init(*_ori_add_output_tensor.allocator(), sub_gather_output_tensor_info);

    _sub_multi_gpt_o_tensor.allocator()->init(*_ori_multi_gpt_o_tensor.allocator(), sub_gather_output_tensor_info);

    _sub_final_ln_o_tensor.allocator()->init(*_ori_final_ln_o_tensor.allocator(), sub_gather_output_tensor_info);

    BITensorShape sub_lm_head_output_tensor_shape(class_num, cur_seq_len, cur_batch_size);
    BITensorInfo sub_lm_head_output_tensor_info(sub_lm_head_output_tensor_shape, 1, BIDataType::F16);
    sub_lm_head_output_tensor_info.set_format(Format::F16);
    _sub_lm_head_output_tensor.allocator()->init(*_ori_lm_head_output_tensor.allocator(), sub_lm_head_output_tensor_info);

    return ret;
}

BIErrCode BIGPT2Model::init_configure_all_layers(const std::vector<int> &tensor_shape) {
    auto ret = BIErrCode::BISuccess;

    int cur_batch_size = 1, cur_seq_len = 1;
    if (tensor_shape.size() >= 1) {
        cur_seq_len = tensor_shape[0];
    }
    if (tensor_shape.size() >= 2) {
        cur_batch_size = tensor_shape[1];
    }

    if ((cur_batch_size > max_batch_size) || (cur_seq_len > max_seq_len)) {
        return BIErrCode::BISizeNotMatch;
    }

    try {
        _gather_layer.configure(&_gather_weight_tensor, &_sub_input_tensor, &_sub_gather_output_tensor, 1);

        _add_layer.configure(&_sub_gather_output_tensor, &_sub_add_weight_tensor, &_sub_add_output_tensor, BIConvertPolicy::SATURATE);

        const BIActivationLayerInfo act_info(BIActivationFunction::GELU);

        gpt_block_config.q_perm = q_perm;
        gpt_block_config.k_perm = k_perm;
        gpt_block_config.qkv_perm = qkv_o_perm;
        gpt_block_config.hidden_size = hidden_size;
        gpt_block_config.max_seq_len = max_seq_len;
        gpt_block_config.max_batch_size = max_batch_size;

        gpt_layer_configs.reserve(layer_num);
        for (int i = 0; i < layer_num; i++) {
            BIIntentGPTLayerConfig layer_config;
            layer_config.ln_1_weight = &_attn_gamma_weight_tensors.at(i);
            layer_config.ln_1_bias = &_attn_gamma_bias_tensors.at(i);
            layer_config.c_attn_weights = &_c_attn_weight_tensors.at(i);
            layer_config.c_attn_bias = &_c_attn_bias_tensors.at(i);
            layer_config.o_attn_weights = &_p_attn_weight_tensors.at(i);
            layer_config.o_attn_bias = &_p_attn_bias_tensors.at(i);
            layer_config.fc_weights = &_c_fc_weight_tensors.at(i);
            layer_config.fc_bias = &_c_fc_bias_tensors.at(i);
            layer_config.ln_2_weight = &_mlp_weight_tensors.at(i);
            layer_config.ln_2_bias = &_mlp_bias_tensors.at(i);
            layer_config.proj_weights = &_c_proj_weight_tensors.at(i);
            layer_config.proj_bias = &_c_proj_bias_tensors.at(i);
            layer_config.act_info = act_info;
            layer_config.layer_idx = i;
            gpt_layer_configs.emplace_back(std::move(layer_config));
        }

        _gpt_multi_block_layer.configure(&_sub_add_output_tensor,
                                            gpt_layer_configs,
                                            gpt_block_config,
                                            cur_batch_size,
                                            cur_seq_len,
                                            &_sub_multi_gpt_o_tensor);

        _final_layernorm_layer.configure(&_sub_multi_gpt_o_tensor,
                                            &_final_layernorm_weight_tensor,
                                            &_final_layernorm_bias_tensor,
                                            &_sub_final_ln_o_tensor);

        GEMMInfo gemm_info = GEMMInfo(false,
                                        false,
                                        true,
                                        false,
                                        false,
                                        false,
                                        BIGEMMLowpOutputStageInfo(),
                                        false,
                                        true,
                                        false,
                                        BIActivationLayerInfo(),
                                        false,
                                        BIWeightFormat::UNSPECIFIED,
                                        false);

        _lm_head_layer.configure(&_sub_final_ln_o_tensor,
                                &_lm_score_weight_tensor,
                                nullptr,
                                &_sub_lm_head_output_tensor,
                                1.0f,
                                1.0f,
                                gemm_info);
    } catch (std::runtime_error &e) {
        ret = BIErrCode::BIGeneralError;
        std::cout << std::string(__FUNCTION__) << " ERROR: " << e.what() << std::endl;
    }

    return ret;
}

BIErrCode BIGPT2Model::dynamic_configure_all_layers(const std::vector<int> &tensor_shape) {
    auto ret = BIErrCode::BISuccess;
    try {
        int cur_batch_size = 1, cur_seq_len = 1;
        if (tensor_shape.size() >= 1) {
            cur_seq_len = tensor_shape[0];
        }
        if (tensor_shape.size() >= 2) {
            cur_batch_size = tensor_shape[1];
        }

        if ((cur_batch_size > max_batch_size) || (cur_seq_len > max_seq_len)) {
            return BIErrCode::BISizeNotMatch;
        }

        _gather_layer.dynamic_configure(&_sub_input_tensor, &_sub_gather_output_tensor);

        _add_layer.dynamic_configure(&_sub_gather_output_tensor, &_sub_add_weight_tensor, true);

        _gpt_multi_block_layer.dynamic_configure(&_sub_add_output_tensor, cur_seq_len, cur_batch_size);

        _final_layernorm_layer.dynamic_configure(&_sub_multi_gpt_o_tensor);

        _lm_head_layer.dynamic_configure();
    } catch (std::runtime_error &e) {
        ret = BIErrCode::BIGeneralError;
        std::cout << std::string(__FUNCTION__) << " ERROR: " << e.what() << std::endl;
    }

    return ret;
}


void BIGPT2Model::print_tensor(const BatmanInfer::BITensor &tensor, const std::string &name , const BatmanInfer::BIIOFormatInfo::PrintRegion region) {
    std::cout << name << std::endl;
    BatmanInfer::BIIOFormatInfo format;
    format.element_delim = ", "; // 元素之间用逗号分隔
    format.row_delim = "\n"; // 每行换行
    format.align_columns = true; // 对齐列
    format.print_region = region;

    tensor.print(std::cout, format);
}
