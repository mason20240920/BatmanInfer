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
    // 创建 kvcache
    int num_head = 12;
    int head_dim = 64;
#ifdef FLOAT_VER
    KVCacheManager::initialize(2048, num_head * head_dim * sizeof(float16_t) * 2), max_seq_len;
#elifdef FIX_VER
    KVCacheManager::initialize(2048, num_head * head_dim * sizeof(int8_t) + num_head * head_dim * sizeof(float16_t), max_seq_len);
#endif
    kv_root_id = KVCacheManager::getInstance().root_id();
}

BIErrCode BIGPT2Model::bi_init(const char *data_in, size_t data_size, std::vector< std::vector<float> > &output_vec, unsigned int &kv_cache_id) {
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
    ret = init_configure_all_layers();
    CHECK_SUCCESS(ret);

    // 为了使内存正确分配，需要在开始的时候 run 一次，以执行每一个算子中的 prepare 函数
    std::vector<std::vector<unsigned int> > tmp_input_vec = {{0}};
    fill_tensor_data_2D(_sub_input_tensor, tmp_input_vec, 1);
    // _output_positions = {1};

    std::vector<size_t> avail_lens{1};
    std::vector<unsigned int> kv_block_ids;
    ret = bi_run(avail_lens, output_vec, kv_block_ids, true);
    kv_cache_id = kv_block_ids[0];
    CHECK_SUCCESS(ret);

    return ret;
}

BIErrCode BIGPT2Model::bi_set_input(std::vector<std::vector<unsigned int> > &input_vec, std::vector< std::vector<unsigned int> > &kv_cache_id_map) {
    auto ret = BIErrCode::BISuccess;

    int cur_batch_size = static_cast<int>(input_vec.size()), cur_seq_len = 0;
    // _output_positions.clear();
    // _output_positions.reserve(cur_batch_size);
    for (auto &item: input_vec) {
        int item_len = static_cast<int>(item.size());
        cur_seq_len = cur_seq_len > item_len ? cur_seq_len : item_len;
        // _output_positions.emplace_back(item_len);
    }

    const std::vector<int> tensor_shape = {cur_seq_len, cur_batch_size};

    // adjust tensor shapes
    ret = set_all_intermediate_tensors(tensor_shape);
    CHECK_SUCCESS(ret);

    // set input value
    ret = fill_tensor_data_2D(_sub_input_tensor, input_vec, cur_seq_len);
    CHECK_SUCCESS(ret);

    // configure some layers dynamically
    ret = dynamic_configure_all_layers(tensor_shape, kv_cache_id_map);
    CHECK_SUCCESS(ret);

    return ret;
}

BIErrCode BIGPT2Model::bi_run(std::vector<size_t> &avail_lens, std::vector<std::vector<float> > &output_vec, std::vector<unsigned int> &kv_block_ids, bool is_init) {
    auto ret = BIErrCode::BISuccess;

    try {
        // print_tensor(_sub_input_tensor, "_sub_input_tensor");

        _gather_layer.run();
        // print_tensor(_sub_gather_output_tensor, "_sub_gather_output_tensor");

        _add_layer.run();
        // print_tensor(_sub_add_output_tensor, "_sub_add_output_tensor");

        if (is_init) {
            _pack.add_tensor(ACL_SRC, &_sub_add_output_tensor);
            _pack.add_tensor(ACL_DST, &_sub_split_add_output_tensor);
        }
        BINEScheduler::get().schedule_kv_split(_pack, avail_lens);
        // print_tensor(_sub_split_add_output_tensor, "_sub_split_add_output_tensor");

        _attn_lowp_layer.set_avail_lens(&avail_lens);
#ifdef FIX_VER
        ret = _attn_lowp_layer.run();
        if (ret != BIErrCode::BISuccess) {
            return ret;
        }
        _attn_lowp_layer.get_kv_block_ids(kv_block_ids);
        // print_tensor(_sub_attn_output_tensor, "_sub_attn_output_tensor");
#elifdef FLOAT_VER
        ret = _attn_layer.run();
        if (ret != BIErrCode::BISuccess) {
            return ret;
        }
        _attn_layer.get_kv_block_ids(kv_block_ids);
        // print_tensor(_sub_attn_output_tensor, "_sub_attn_output_tensor");
#endif

        _attn_rms_add_layer.run();
        // print_tensor(_sub_mlp_input_tensor, "_sub_mlp_input_tensor");

        _mlp_layer.run();
        // print_tensor(_sub_mlp_output_tensor, "_sub_mlp_output_tensor");

        _add_mlp_layer.run();
        // print_tensor(_sub_add_mlp_output_tensor, "_sub_add_mlp_output_tensor");

        _rms_norm_layer.run();
        // print_tensor(_sub_mlp_rms_output_tensor, "_sub_mlp_rms_output_tensor");

        _lm_head_layer.run();
        // print_tensor(_sub_lm_head_output_tensor, "_sub_lm_head_output_tensor");
    } catch (std::runtime_error &e) {
        ret = BIErrCode::BIGeneralError;
        std::cout << std::string(__FUNCTION__) << " ERROR: " << e.what() << std::endl;
    }
    CHECK_SUCCESS(ret);

    // fill output_vec
    output_vec.clear();
    const int cur_seq_len = static_cast<int>(_sub_lm_head_output_tensor.info()->tensor_shape()[BIWindow::DimY]);
    const size_t cur_batch_size = _sub_lm_head_output_tensor.info()->tensor_shape()[BIWindow::DimZ];

    // if (_output_positions.size() != cur_batch_size) {
    //     return BIErrCode::BISizeNotMatch;
    // }

    output_vec.resize(cur_batch_size);
    const auto result_ptr = reinterpret_cast<half *>(_sub_lm_head_output_tensor.buffer());
    // for (size_t item = 0; item < _output_positions.size(); ++item) {
    //     if (_output_positions[item] > cur_seq_len) {
    //         output_vec.clear();
    //         return BIErrCode::BISizeNotMatch;
    //     }
    //     output_vec[item].reserve(dict_size);
    //
    //     const auto cur_result_ptr = result_ptr + (item * cur_seq_len + _output_positions[item] - 1) * dict_size;
    //
    //     for (size_t prob_idx = 0; prob_idx < dict_size; ++prob_idx) {
    //         output_vec[item].push_back(static_cast<float>(cur_result_ptr[prob_idx]));
    //     }
    // }
    for (size_t item = 0; item < cur_batch_size; ++item) {
        output_vec[item].reserve(dict_size);

        const auto cur_result_ptr = result_ptr + item * dict_size;

        for (size_t prob_idx = 0; prob_idx < dict_size; ++prob_idx) {
            output_vec[item].push_back(static_cast<float>(cur_result_ptr[prob_idx]));
        }
    }

    return ret;
}

bool BIGPT2Model::bi_valid_decode_ids(std::vector<unsigned int> &kv_block_ids) {
    return KVCacheManager::getInstance().vaild_decode_ids(kv_block_ids);
}

BIErrCode BIGPT2Model::bi_release_kvcache_block(std::vector<unsigned int> &kv_block_ids) {
    return KVCacheManager::getInstance().release_useless_decodes_id(kv_block_ids);
}

BIErrCode BIGPT2Model::bi_release_kvcache_leaf_block(std::vector<unsigned int> &kv_block_ids) {
    return KVCacheManager::getInstance().release_end_symbol(kv_block_ids);
}

void BIGPT2Model::bi_get_avaliable_kvblock_count(unsigned int &avaliable_kvblock_count) {
    avaliable_kvblock_count = KVCacheManager::getInstance().get_avaliable_block_count();
}

BIErrCode BIGPT2Model::bi_reset(unsigned int &kv_cache_id) {
    auto ret = BIErrCode::BISuccess;

    KVCacheManager::getInstance().reset_decode_lst();
    kv_cache_id = KVCacheManager::getInstance().root_id();

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

BIErrCode BIGPT2Model::load_weight_tensor(BITensor &tensor, GPT2ResOrder res_order, OrderPtrMap &order2ptr, bool need_transpose) {
    if (order2ptr.find(res_order) == order2ptr.end()) {
        return BIErrCode::BIResNotExists;
    }

    char *tmp_ptr = order2ptr[res_order];

    auto header = reinterpret_cast<GPT2ResHeader *>(tmp_ptr);

    // 如果当前张量需要进行转置，需要先将数据接收张量进行转置(当前只有二维张量进行转置，且需要转置的张量数据为 f16类型)，数据接收完毕后还需要将张量转置回去
    BITensor tensor_tmp;
    if (need_transpose) {
        auto tensor_num_dim = tensor.info()->num_dimensions();
        if (tensor_num_dim != 2) {
            return BIErrCode::BIResDamaged;
        }

        tensor_tmp.allocator()->init(BITensorInfo(BITensorShape(tensor.info()->tensor_shape()[1], tensor.info()->tensor_shape()[0]), 1, tensor.info()->data_type()));
        tensor_tmp.allocator()->allocate();

        // 检查 shape是否对得上
        for (size_t i = 0; i < tensor_tmp.info()->num_dimensions(); ++i) {
            if (tensor_tmp.info()->tensor_shape()[i] != header->shape[i]) {
                return BIErrCode::BIResDamaged;
            }
        }
        for (auto i = tensor_tmp.info()->num_dimensions(); i < tensor_max_dim; ++i) {
            if (header->shape[i] != 1) {
                return BIErrCode::BIResDamaged;
            }
        }

        // shape 检查完，其实这一步可以不用再检查了，但是保险起见还是再检查一遍
        if (tensor_tmp.info()->total_size() != header->data_length) {
            return BIErrCode::BIResDamaged;
        }

        // 检查类型是否对得上
        std::string tensor_type_str = BatmanInfer::utils::get_typestring(tensor_tmp.info()->data_type());
        if (tensor_type_str != std::string(header->data_type)) {
            return BIErrCode::BIResDamaged;
        }

        // 拷贝具体数据
        tmp_ptr += sizeof(GPT2ResHeader);
        memcpy(reinterpret_cast<char *>(tensor_tmp.buffer()), tmp_ptr, header->data_length);

        // 数据拷贝完成，需要将数据进行转置
        BINETranspose transpose;
        transpose.configure(&tensor_tmp, &tensor);
        transpose.run();
    }
    else {
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
    }

    return BIErrCode::BISuccess;
}

BIErrCode BIGPT2Model::load_scale_vector(std::vector<float> &scales, GPT2ResOrder res_order, OrderPtrMap &order2ptr) {
    if (order2ptr.find(res_order) == order2ptr.end()) {
        return BIErrCode::BIResNotExists;
    }

    char *tmp_ptr = order2ptr[res_order];

    auto header = reinterpret_cast<GPT2ResHeader *>(tmp_ptr);

    auto float_ptr = reinterpret_cast<float *>(tmp_ptr + sizeof(GPT2ResHeader));

    scales.reserve(header->shape[0]);
    for (size_t i = 0; i < header->shape[0]; ++i) {
        scales.push_back(float_ptr[i]);
    }

    return BIErrCode::BISuccess;
}

BIErrCode BIGPT2Model::load_hyper_params(OrderPtrMap &order2ptr) {
#ifdef FIX_VER
    if (order2ptr.find(GPT2ResOrder::decode_layer_scales) == order2ptr.end()) {
        return BIErrCode::BIResNotExists;
    }
    char *tmp_ptr = order2ptr[GPT2ResOrder::decode_layer_scales];

    const auto header = reinterpret_cast<GPT2ResHeader *>(tmp_ptr);
    if (header->data_length != sizeof(HyperParameters)) {
        return BIErrCode::BIResDamaged;
    }

    const auto all_param = reinterpret_cast<HyperParameters *>(tmp_ptr + sizeof(GPT2ResHeader));

    _attn_hyper_params.attn_gemm_i_scale     = all_param->attn_input_scale;
    _attn_hyper_params.attn_gemm_i_zero      = all_param->attn_input_zp;

    _attn_hyper_params.attn_gemm_o_scale     = all_param->attn_output_scale;
    _attn_hyper_params.attn_gemm_o_zero      = all_param->attn_output_zp;

    _attn_hyper_params.query_scale           = all_param->q_output_scale;
    _attn_hyper_params.query_zp              = all_param->q_output_zp;

    _attn_hyper_params.key_scale             = all_param->k_output_scale;
    _attn_hyper_params.key_zp                = all_param->k_output_zp;

    _attn_hyper_params.value_scale           = all_param->v_output_scale;
    _attn_hyper_params.value_zp              = all_param->v_output_zp;

    _attn_hyper_params.proj_in_scale         = all_param->out_input_scale;
    _attn_hyper_params.proj_in_zp            = all_param->out_input_zp;

    _mlp_hyper_params.fc1_input_scale        = all_param->fc1_input_scale;
    _mlp_hyper_params.fc1_input_zero_point  = all_param->fc1_input_zp;

    _mlp_hyper_params.fc1_output_scale       = all_param->fc1_output_scale;
    _mlp_hyper_params.fc1_output_zero_point  = all_param->fc1_output_zp;

    _mlp_hyper_params.gelu_output_scale      = all_param->fc2_input_scale;
    _mlp_hyper_params.gelu_output_zero_point = all_param->fc2_input_zp;
#endif

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

    const BITensorShape ori_attn_output_tensor_shape(hidden_size, 1, max_batch_size);
    _ori_attn_rms_output_tensor.allocator()->init(BITensorInfo(ori_attn_output_tensor_shape, 1, BIDataType::F16));

    const BITensorShape add_weight_tensor_shape(hidden_size, max_seq_len);
    _add_weight_tensor.allocator()->init(BITensorInfo(add_weight_tensor_shape, 1, BIDataType::F16));

    _ori_add_output_tensor.allocator()->init(BITensorInfo(ori_gather_output_tensor_shape, 1, BIDataType::F16));

    _ori_split_add_output_tensor.allocator()->init(BITensorInfo(ori_attn_output_tensor_shape, 1, BIDataType::F16));

    const BITensorShape attn_gamma_weight_tensor_shape(hidden_size);
    _attn_gamma_weight_tensor.allocator()->init(BITensorInfo(attn_gamma_weight_tensor_shape, 1, BIDataType::F16));

    _ori_attn_output_tensor.allocator()->init(BITensorInfo(ori_gather_output_tensor_shape, 1, BIDataType::F16));

    const BITensorShape attn_qkv_weight_tensor_shape(hidden_size * 3, hidden_size);
#ifdef FLOAT_VER
    _attn_qkv_weight_tensor.allocator()->init(BITensorInfo(attn_qkv_weight_tensor_shape, 1, BIDataType::F16));
#elifdef FIX_VER
    _attn_qkv_weight_tensor.allocator()->init(BITensorInfo(attn_qkv_weight_tensor_shape, 1, BIDataType::QSYMM8_PER_CHANNEL));
#endif

    const BITensorShape attn_qkv_bias_tensor_shape(hidden_size * 3);
#ifdef FLOAT_VER
    _attn_qkv_bias_tensor.allocator()->init(BITensorInfo(attn_qkv_bias_tensor_shape, 1, BIDataType::F16));
#elifdef FIX_VER
    _attn_qkv_bias_tensor.allocator()->init(BITensorInfo(attn_qkv_bias_tensor_shape, 1, BIDataType::S32));
#endif

    const BITensorShape attn_c_proj_weight_tensor_shape(hidden_size, hidden_size);
    _attn_c_proj_weight_tensor.allocator()->init(BITensorInfo(attn_c_proj_weight_tensor_shape, 1, BIDataType::F16));

    const BITensorShape attn_c_proj_bias_tensor_shape(hidden_size);
    _attn_c_proj_bias_tensor.allocator()->init(BITensorInfo(attn_c_proj_bias_tensor_shape, 1, BIDataType::F16));

    const BITensorShape mlp_gamma_tensor_shape(hidden_size);
    _mlp_gamma_weight_tensor.allocator()->init(BITensorInfo(mlp_gamma_tensor_shape, 1, BIDataType::F16));

    const BITensorShape c_fc_weight_tensor_shape(hidden_size * 4, hidden_size);
#ifdef FLOAT_VER
    _c_fc_weight_tensor.allocator()->init(BITensorInfo(c_fc_weight_tensor_shape, 1, BIDataType::F16));
#elifdef FIX_VER
    _c_fc_weight_tensor.allocator()->init(BITensorInfo(c_fc_weight_tensor_shape, 1, BIDataType::QSYMM8_PER_CHANNEL));
#endif

    const BITensorShape c_fc_bias_tensor_shape(hidden_size * 4);
#ifdef FLOAT_VER
    _c_fc_bias_tensor.allocator()->init(BITensorInfo(c_fc_bias_tensor_shape, 1, BIDataType::F16));
#elifdef FIX_VER
    _c_fc_bias_tensor.allocator()->init(BITensorInfo(c_fc_bias_tensor_shape, 1, BIDataType::S32));
#endif

    const BITensorShape ori_mlp_outpu_tensor_shape(hidden_size, 1, max_batch_size);
    _ori_mlp_output_tensor.allocator()->init(BITensorInfo(ori_mlp_outpu_tensor_shape, 1, BIDataType::F16));

    const BITensorShape c_proj_weight_tensor_shape(hidden_size, hidden_size * 4);
    _c_proj_weight_tensor.allocator()->init(BITensorInfo(c_proj_weight_tensor_shape, 1, BIDataType::F16));

    const BITensorShape c_proj_bias_tensor_shape(hidden_size);
    _c_proj_bias_tensor.allocator()->init(BITensorInfo(c_proj_bias_tensor_shape, 1, BIDataType::F16));

    const BITensorShape ori_add_mlp_output_tensor_shape(hidden_size, 1, max_batch_size);
    _ori_add_mlp_output_tensor.allocator()->init(BITensorInfo(ori_add_mlp_output_tensor_shape, 1, BIDataType::F16));

    const BITensorShape rms_gamma_weight_tensor_shape(hidden_size);
    _rms_gamma_weight_tensor.allocator()->init(BITensorInfo(rms_gamma_weight_tensor_shape, 1, BIDataType::F16));

    _ori_mlp_rms_output_tensor.allocator()->init(BITensorInfo(ori_add_mlp_output_tensor_shape, 1, BIDataType::F16));

    const BITensorShape lm_head_weight_tensor_shape(dict_size, hidden_size);
    _lm_head_weight_tensor.allocator()->init(BITensorInfo(lm_head_weight_tensor_shape, 1, BIDataType::F16));

    const BITensorShape ori_lm_head_output_tensor_shape(dict_size, 1, max_batch_size);
    _ori_lm_head_output_tensor.allocator()->init(BITensorInfo(ori_lm_head_output_tensor_shape, 1, BIDataType::F16));

    const BITensorShape eos_qkv_smooth_o_tensor_shape(64, 12, 16);
    _eos_k_smooth_o_tensor.allocator()->init(BITensorInfo(eos_qkv_smooth_o_tensor_shape, 1, BIDataType::F16));

    _eos_q_smooth_o_tensor.allocator()->init(BITensorInfo(eos_qkv_smooth_o_tensor_shape, 1, BIDataType::F16));

    _eos_v_smooth_o_tensor.allocator()->init(BITensorInfo(eos_qkv_smooth_o_tensor_shape, 1, BIDataType::QSYMM8_PER_CHANNEL));

    // 内存统一管理
    _memory_group.manage(&_ori_input_tensor);
    _memory_group.manage(&_gather_weight_tensor);
    _memory_group.manage(&_ori_gather_output_tensor);
    _memory_group.manage(&_ori_attn_rms_output_tensor);
    _memory_group.manage(&_add_weight_tensor);
    _memory_group.manage(&_ori_add_output_tensor);
    _memory_group.manage(&_ori_split_add_output_tensor);
    _memory_group.manage(&_attn_gamma_weight_tensor);
    _memory_group.manage(&_ori_attn_output_tensor);
    _memory_group.manage(&_attn_qkv_weight_tensor);
    _memory_group.manage(&_attn_qkv_bias_tensor);
    _memory_group.manage(&_attn_c_proj_weight_tensor);
    _memory_group.manage(&_attn_c_proj_bias_tensor);
    _memory_group.manage(&_mlp_gamma_weight_tensor);
    _memory_group.manage(&_c_fc_weight_tensor);
    _memory_group.manage(&_c_fc_bias_tensor);
    _memory_group.manage(&_ori_mlp_output_tensor);
    _memory_group.manage(&_c_proj_weight_tensor);
    _memory_group.manage(&_c_proj_bias_tensor);
    _memory_group.manage(&_ori_add_mlp_output_tensor);
    _memory_group.manage(&_rms_gamma_weight_tensor);
    _memory_group.manage(&_ori_mlp_rms_output_tensor);
    _memory_group.manage(&_lm_head_weight_tensor);
    _memory_group.manage(&_ori_lm_head_output_tensor);
    _memory_group.manage(&_eos_k_smooth_o_tensor);
    _memory_group.manage(&_eos_q_smooth_o_tensor);
    _memory_group.manage(&_eos_v_smooth_o_tensor);

    _ori_input_tensor.allocator()->allocate();
    _gather_weight_tensor.allocator()->allocate();
    _ori_gather_output_tensor.allocator()->allocate();
    _ori_attn_rms_output_tensor.allocator()->allocate();
    _add_weight_tensor.allocator()->allocate();
    _ori_add_output_tensor.allocator()->allocate();
    _ori_split_add_output_tensor.allocator()->allocate();
    _attn_gamma_weight_tensor.allocator()->allocate();
    _ori_attn_output_tensor.allocator()->allocate();
    _attn_qkv_weight_tensor.allocator()->allocate();
    _attn_qkv_bias_tensor.allocator()->allocate();
    _attn_c_proj_weight_tensor.allocator()->allocate();
    _attn_c_proj_bias_tensor.allocator()->allocate();
    _mlp_gamma_weight_tensor.allocator()->allocate();
    _c_fc_weight_tensor.allocator()->allocate();
    _c_fc_bias_tensor.allocator()->allocate();
    _ori_mlp_output_tensor.allocator()->allocate();
    _c_proj_weight_tensor.allocator()->allocate();
    _c_proj_bias_tensor.allocator()->allocate();
    _ori_add_mlp_output_tensor.allocator()->allocate();
    _rms_gamma_weight_tensor.allocator()->allocate();
    _ori_mlp_rms_output_tensor.allocator()->allocate();
    _lm_head_weight_tensor.allocator()->allocate();
    _ori_lm_head_output_tensor.allocator()->allocate();
    _eos_k_smooth_o_tensor.allocator()->allocate();
    _eos_q_smooth_o_tensor.allocator()->allocate();
    _eos_v_smooth_o_tensor.allocator()->allocate();

    _scope_manager = std::make_unique<BIMemoryGroupResourceScope>(_memory_group);

    // load gather weight
    ret = load_weight_tensor(_gather_weight_tensor, GPT2ResOrder::gather_weight, order2ptr, false);
    CHECK_SUCCESS(ret);

    // load add weight
    ret = load_weight_tensor(_add_weight_tensor, GPT2ResOrder::add_weight, order2ptr, false);
    CHECK_SUCCESS(ret);

    // load gamma weight
    ret = load_weight_tensor(_attn_gamma_weight_tensor, GPT2ResOrder::attn_gamma_weight, order2ptr, false);
    CHECK_SUCCESS(ret);

    // load attn qkiv weight
    ret = load_weight_tensor(_attn_qkv_weight_tensor, GPT2ResOrder::attn_qkv_weight, order2ptr, false);
    CHECK_SUCCESS(ret);
#ifdef FIX_VER
    // load attn qkv scales
    std::vector<float> attn_qkv_scales;
    ret = load_scale_vector(attn_qkv_scales, GPT2ResOrder::attn_qkv_weight_scale, order2ptr);
    CHECK_SUCCESS(ret);
    _attn_qkv_weight_tensor.info()->set_quantization_info(attn_qkv_scales);
#endif

    // load qkv bias
    ret = load_weight_tensor(_attn_qkv_bias_tensor, GPT2ResOrder::attn_qkv_bias, order2ptr, false);
    CHECK_SUCCESS(ret);

    // load attn c proj weight
    ret = load_weight_tensor(_attn_c_proj_weight_tensor, GPT2ResOrder::attn_c_proj_weight, order2ptr, false);
    CHECK_SUCCESS(ret);

    // load attn c proj bias
    ret = load_weight_tensor(_attn_c_proj_bias_tensor, GPT2ResOrder::attn_c_proj_bias, order2ptr, false);
    CHECK_SUCCESS(ret);

    // load mlp gamma weight
    ret = load_weight_tensor(_mlp_gamma_weight_tensor, GPT2ResOrder::mlp_gamma_weight, order2ptr, false);
    CHECK_SUCCESS(ret);

    // load c fc weight
    ret = load_weight_tensor(_c_fc_weight_tensor, GPT2ResOrder::c_fc_weight, order2ptr, false);
    CHECK_SUCCESS(ret);
#ifdef FIX_VER
    std::vector<float> c_fc_weight_scales;
    // load c fc weight scales
    ret = load_scale_vector(c_fc_weight_scales, GPT2ResOrder::c_fc_weight_scale, order2ptr);
    CHECK_SUCCESS(ret);

    _c_fc_weight_q_info = BIQuantizationInfo(c_fc_weight_scales);
    _c_fc_weight_tensor.info()->set_quantization_info(_c_fc_weight_q_info);
#endif

    // load c fc bias
    ret = load_weight_tensor(_c_fc_bias_tensor, GPT2ResOrder::c_fc_bias, order2ptr, false);
    CHECK_SUCCESS(ret);

    // load c proj weight
    ret = load_weight_tensor(_c_proj_weight_tensor, GPT2ResOrder::c_proj_weight, order2ptr, false);
    CHECK_SUCCESS(ret);

    // load c proj bias
    ret = load_weight_tensor(_c_proj_bias_tensor, GPT2ResOrder::c_proj_bias, order2ptr, false);
    CHECK_SUCCESS(ret);

    // load rms gamma weight
    ret = load_weight_tensor(_rms_gamma_weight_tensor, GPT2ResOrder::rms_gamma_weight, order2ptr, false);
    CHECK_SUCCESS(ret);

    // load lm head weight (lm head weight 和 gather_weight 数据相同，但需要进行转置)
    ret = load_weight_tensor(_lm_head_weight_tensor, GPT2ResOrder::gather_weight, order2ptr, true);
    CHECK_SUCCESS(ret);

    // load eos_k_smooth_o weight
    ret = load_weight_tensor(_eos_k_smooth_o_tensor, GPT2ResOrder::eos_k_smooth_o, order2ptr, false);
    CHECK_SUCCESS(ret);
    KVCacheManager::getInstance().memcpy_init_eos_buffer(_eos_k_smooth_o_tensor.buffer(), max_seq_len - 1, max_seq_len, true, true);

    // load eos_q_smooth_o weight
    ret = load_weight_tensor(_eos_q_smooth_o_tensor, GPT2ResOrder::eos_q_smooth_o, order2ptr, false);
    CHECK_SUCCESS(ret);

    // load eos_v_smooth_o weight
    ret = load_weight_tensor(_eos_v_smooth_o_tensor, GPT2ResOrder::eos_v_smooth_o, order2ptr, false);
    CHECK_SUCCESS(ret);
    KVCacheManager::getInstance().memcpy_init_eos_buffer(_eos_v_smooth_o_tensor.buffer(), max_seq_len - 1, max_seq_len, false, true);

#ifdef FIX_VER
    // load hyper parameters
    ret = load_hyper_params(order2ptr);
    CHECK_SUCCESS(ret);
#endif

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

    BITensorShape sub_attn_output_tensor_shape(hidden_size, 1, cur_batch_size);
    BITensorInfo sub_attn_output_tensor_info(sub_attn_output_tensor_shape, 1, BIDataType::F16);
    sub_attn_output_tensor_info.set_format(Format::F16);
    _sub_split_add_output_tensor.allocator()->init(*_ori_split_add_output_tensor.allocator(), sub_attn_output_tensor_info);

    _sub_attn_output_tensor.allocator()->init(*_ori_attn_output_tensor.allocator(), sub_attn_output_tensor_info);

    _sub_mlp_input_tensor.allocator()->init(*_ori_attn_rms_output_tensor.allocator(), sub_attn_output_tensor_info);

    _sub_mlp_output_tensor.allocator()->init(*_ori_mlp_output_tensor.allocator(), sub_attn_output_tensor_info);

    _sub_add_mlp_output_tensor.allocator()->
            init(*_ori_add_mlp_output_tensor.allocator(), sub_attn_output_tensor_info);

    _sub_mlp_rms_output_tensor.allocator()->
            init(*_ori_mlp_rms_output_tensor.allocator(), sub_attn_output_tensor_info);

    BITensorShape sub_lm_head_output_tensor_shape(dict_size, 1, cur_batch_size);
    BITensorInfo sub_lm_head_output_tensor_info(sub_lm_head_output_tensor_shape, 1, BIDataType::F16);
    sub_lm_head_output_tensor_info.set_format(Format::F16);
    _sub_lm_head_output_tensor.allocator()->init(*_ori_lm_head_output_tensor.allocator(),
                                                 sub_lm_head_output_tensor_info);

    return ret;
}

BIErrCode BIGPT2Model::init_configure_all_layers() {
    auto ret = BIErrCode::BISuccess;
    try {
        _gather_layer.configure(&_gather_weight_tensor, &_sub_input_tensor, &_sub_gather_output_tensor, 1);

        _add_layer.configure(&_sub_gather_output_tensor, &_sub_add_weight_tensor,
                             &_sub_add_output_tensor, BIConvertPolicy::SATURATE);

#ifdef FIX_VER
        _attn_lowp_layer.configure(&_sub_split_add_output_tensor,
                                   &_attn_gamma_weight_tensor,
                                   &_attn_qkv_weight_tensor,
                                   &_attn_qkv_bias_tensor,
                                   &_attn_c_proj_weight_tensor,
                                   &_attn_c_proj_bias_tensor,
                                   &_eos_q_smooth_o_tensor,
                                   _attn_hyper_params.attn_gemm_i_scale,
                                   _attn_hyper_params.attn_gemm_i_zero,
                                   _attn_hyper_params.attn_gemm_o_scale,
                                   _attn_hyper_params.attn_gemm_o_zero,
                                   _attn_hyper_params.query_scale,
                                   _attn_hyper_params.query_zp,
                                   _attn_hyper_params.value_scale,
                                   _attn_hyper_params.value_zp,
                                   _attn_hyper_params.key_scale,
                                   _attn_hyper_params.key_zp,
                                   AttnHyperParams::softmax_q_scale,
                                   AttnHyperParams::softmax_zp,
                                   _attn_hyper_params.proj_in_scale,
                                   _attn_hyper_params.proj_in_zp,
                                   q_perm,
                                   k_perm,
                                   qkv_o_perm,
                                   hidden_size,
                                   max_seq_len,
                                   max_batch_size,
                                   &_sub_attn_output_tensor);
#elifdef FLOAT_VER
        _attn_layer.configure(&_sub_split_add_output_tensor,
                              &_attn_gamma_weight_tensor,
                              &_attn_qkv_weight_tensor,
                              &_attn_qkv_bias_tensor,
                              &_attn_c_proj_weight_tensor,
                              &_attn_c_proj_bias_tensor,
                              &_eos_q_smooth_o_tensor,
                              q_perm,
                              k_perm,
                              qkv_o_perm,
                              hidden_size,
                              max_seq_len,
                              max_batch_size,
                              &_sub_attn_output_tensor);
#endif

        _attn_rms_add_layer.configure(&_sub_split_add_output_tensor, &_sub_attn_output_tensor,
                                      &_sub_mlp_input_tensor, BIConvertPolicy::SATURATE);

#ifdef FIX_VER
        _mlp_layer.configure(&_sub_mlp_input_tensor,
                             _mlp_hyper_params.fc1_input_scale,
                             _mlp_hyper_params.fc1_input_zero_point,
                             &_c_fc_weight_tensor,
                             &_c_fc_bias_tensor,
                             &_c_fc_weight_q_info,
                             _mlp_hyper_params.fc1_output_scale,
                             _mlp_hyper_params.fc1_output_zero_point,
                             _mlp_hyper_params.gelu_output_scale,
                             _mlp_hyper_params.gelu_output_zero_point,
                             &_c_proj_weight_tensor,
                             &_c_proj_bias_tensor,
                             &_mlp_gamma_weight_tensor,
                             &_sub_mlp_output_tensor,
                             max_batch_size,
                             1);
#elifdef FLOAT_VER
        const BIActivationLayerInfo act_info(BIActivationFunction::GELU);
        _mlp_layer.configure(&_sub_mlp_input_tensor,
                             &_c_fc_weight_tensor,
                             &_c_fc_bias_tensor,
                             &_c_proj_weight_tensor,
                             &_c_proj_bias_tensor,
                             &_mlp_gamma_weight_tensor,
                             act_info,
                             &_sub_mlp_output_tensor,
                             max_batch_size,
                             1);
#endif

        _add_mlp_layer.configure(&_sub_mlp_output_tensor, &_sub_mlp_input_tensor,
                                 &_sub_add_mlp_output_tensor, BIConvertPolicy::SATURATE);

        _rms_norm_layer.configure(&_sub_add_mlp_output_tensor, &_rms_gamma_weight_tensor, &_sub_mlp_rms_output_tensor);

        const auto gemm_info = GEMMInfo(false,
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

        _lm_head_layer.configure(&_sub_mlp_rms_output_tensor,
                                 &_lm_head_weight_tensor,
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

BIErrCode BIGPT2Model::dynamic_configure_all_layers(const std::vector<int> &tensor_shape, std::vector< std::vector<unsigned int> > &kv_cache_id_map) {
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

#ifdef FIX_VER
        _attn_lowp_layer.dynamic_configure(&_sub_split_add_output_tensor, cur_seq_len, cur_batch_size, kv_cache_id_map);
#elifdef FLOAT_VER
        _attn_layer.dynamic_configure(&_sub_split_add_output_tensor, cur_seq_len, cur_batch_size, kv_cache_id_map);
#endif

        _attn_rms_add_layer.dynamic_configure(&_sub_split_add_output_tensor, &_sub_attn_output_tensor, true);

        _mlp_layer.dynamic_configure(&_sub_mlp_input_tensor, cur_batch_size);

        _add_mlp_layer.dynamic_configure(&_sub_mlp_output_tensor, &_sub_mlp_input_tensor, false);

        _rms_norm_layer.dynamic_configure(&_sub_add_mlp_output_tensor);

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
