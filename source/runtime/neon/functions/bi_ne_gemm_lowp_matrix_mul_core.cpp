//
// Created by Mason on 2025/1/21.
//

#include <runtime/neon/functions/bi_ne_gemm_lowp_matrix_mul_core.hpp>

#include <data/core/bi_i_tensor.hpp>
#include <data/core/bi_utils.hpp>
#include <data/core/utils/data_type_utils.hpp>
#include <data/core/utils/quantization/asymm_helpers.hpp>
#include <data/core/bi_vlidate.hpp>
#include <runtime/bi_i_weights_manager.hpp>
#include <runtime/bi_memory_group.hpp>
#include <runtime/bi_tensor.hpp>

#include <data/core/helpers/bi_memory_helpers.hpp>
#include <cpu/operators/bi_cpu_gemm_lowp_matrix_multiply_core.hpp>

#include <set>
#include <utility>

using namespace BatmanInfer::experimental;

namespace BatmanInfer {
    struct BINEGEMMLowpMatrixMultipleCore::Impl {
        const BIITensor *b{nullptr};
        std::unique_ptr<cpu::BICpuGemmLowpMatrixMultiplyCore> op{nullptr};
        BIITensorPack run_pack{};
        BIITensorPack prep_pack{};
        BIMemoryGroup memory_group{};
        BIIWeightsManager *weights_manager{nullptr};
        BIMemoryRequirements aux_mem_req{};
        WorkspaceData<BITensor> workspace_tensors{};
        BIActivationLayerInfo act_info{};
        bool is_prepared{false};
    };

    BINEGEMMLowpMatrixMultipleCore::BINEGEMMLowpMatrixMultipleCore(std::shared_ptr<BIIMemoryManager> memory_manager,
                                                                   BatmanInfer::BIIWeightsManager *weights_manager)
        : _impl(std::make_unique<Impl>()) {
        _impl->weights_manager = weights_manager;
        _impl->memory_group = BIMemoryGroup(memory_manager);
    }

    BINEGEMMLowpMatrixMultipleCore::~BINEGEMMLowpMatrixMultipleCore() = default;

    void BINEGEMMLowpMatrixMultipleCore::configure(const BatmanInfer::BIITensor *a, const BatmanInfer::BIITensor *b,
                                                   const BatmanInfer::BIITensor *c,
                                                   BatmanInfer::BIITensor *output,
                                                   const BatmanInfer::GEMMInfo &gemm_info) {
        BI_COMPUTE_ERROR_ON_NULLPTR(a, b, output);

        // Make the B matrix dynamic values.
        auto b_info_to_use = b->info()->clone(); // 创建b的一个智能指针
        if (!gemm_info.reshape_b_only_on_first_run())
            b_info_to_use->set_are_values_constant(false);

        _impl->is_prepared = false;
        _impl->memory_group.mappings().clear(); // 清空内存管理
        _impl->b = b;
        _impl->op = std::make_unique<cpu::BICpuGemmLowpMatrixMultiplyCore>();
        _impl->op->configure(a->info(), b_info_to_use.get(), (c != nullptr ? c->info() : nullptr), output->info(),
                             gemm_info);
        _impl->run_pack = {
            {BITensorType::ACL_SRC_0, a},
            {BITensorType::ACL_SRC_1, b},
            {BITensorType::ACL_SRC_2, c},
            {BITensorType::ACL_DST, output}
        };
        _impl->prep_pack = {
            {BITensorType::ACL_SRC_1, b},
            {BITensorType::ACL_SRC_2, c}
        };
        _impl->aux_mem_req = _impl->op->workspace();
        _impl->act_info = gemm_info.activation_info();
        _impl->workspace_tensors = manage_workspace<BITensor>(_impl->aux_mem_req, _impl->memory_group, _impl->run_pack,
                                                              _impl->prep_pack, /* allocate_now */ false);
    }

    void BINEGEMMLowpMatrixMultipleCore::dynamic_configure(const BIITensor *input, const BIITensor *output) const {
        auto b = _impl->run_pack.get_const_tensor(BITensorType::ACL_SRC_1);
        _impl->op->dynamic_configure(input->info(), b->info(), output->info());
    }


    BIStatus
    BINEGEMMLowpMatrixMultipleCore::validate(const BatmanInfer::BIITensorInfo *a, const BatmanInfer::BIITensorInfo *b,
                                             const BatmanInfer::BIITensorInfo *c,
                                             const BatmanInfer::BIITensorInfo *output,
                                             const BatmanInfer::GEMMInfo &gemm_info) {
        // Make the B matrix dynamic values.
        auto b_info_to_use = b->clone();
        if (!gemm_info.reshape_b_only_on_first_run()) {
            b_info_to_use->set_are_values_constant(false);
        }

        return cpu::BICpuGemmLowpMatrixMultiplyCore::validate(a, b_info_to_use.get(), c, output, gemm_info);
    }

    void BINEGEMMLowpMatrixMultipleCore::update_quantization_parameters() {
        // Supported activations in GEMM
        const std::set<BIActivationLayerInfo::ActivationFunction> supported_acts = {
            BIActivationLayerInfo::ActivationFunction::RELU,
            BIActivationLayerInfo::ActivationFunction::BOUNDED_RELU,
            BIActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU
        };

        auto src = _impl->run_pack.get_const_tensor(ACL_SRC_0);
        auto wei = _impl->run_pack.get_const_tensor(ACL_SRC_1);
        auto dst = _impl->run_pack.get_tensor(ACL_DST);

        const BIQuantizationInfo iqinfo = src->info()->quantization_info();
        const BIQuantizationInfo wqinfo = wei->info()->quantization_info();
        const BIQuantizationInfo oqinfo = (dst->info()->total_size() == 0) ? iqinfo : dst->info()->quantization_info();

        BIPixelValue type_min{};
        BIPixelValue type_max{};
        const BIDataType data_type = src->info()->data_type();
        std::tie(type_min, type_max) = get_min_max(data_type);
        auto min_activation = type_min.get<int32_t>();
        auto max_activation = type_max.get<int32_t>();

        const BIUniformQuantizationInfo uoqinfo = oqinfo.uniform();
        if (supported_acts.find(_impl->act_info.activation()) != supported_acts.end()) {
            std::tie(min_activation, max_activation) =
                    get_quantized_activation_min_max(_impl->act_info, data_type, uoqinfo);
        }

        BIGEMMLowpOutputStageInfo output_info;
        output_info.type = BIGEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT;
        output_info.gemmlowp_offset = uoqinfo.offset;
        output_info.gemmlowp_min_bound = min_activation;
        output_info.gemmlowp_max_bound = max_activation;
        output_info.is_quantized_per_channel = false;
        output_info.output_data_type = dst->info()->data_type();
        quantization::calculate_quantized_multipliers(iqinfo, wqinfo, oqinfo, output_info);

        _impl->op->update_quantization_parameters(output_info, src->info()->quantization_info(),
                                                  wei->info()->quantization_info(), true, true);
    }

    void BINEGEMMLowpMatrixMultipleCore::run() {
        prepare();
        BIMemoryGroupResourceScope scope_mg(_impl->memory_group);
        _impl->op->run(_impl->run_pack);
    }

    void BINEGEMMLowpMatrixMultipleCore::prepare() {
        if (!_impl->is_prepared) {
            allocate_tensors(_impl->aux_mem_req, _impl->workspace_tensors);
            _impl->op->prepare(_impl->prep_pack);

            auto has_reshape = std::find_if(_impl->aux_mem_req.begin(), _impl->aux_mem_req.end(),
                                            [](const BIMemoryInfo &m) -> bool {
                                                return m.lifetime == MemoryLifetime::Persistent;
                                            });

            if (has_reshape != std::end(_impl->aux_mem_req))
                _impl->b->mark_as_unused();

            // Release temporary tensors that are only used in prepare stage
            release_temporaries<BITensor>(_impl->aux_mem_req, _impl->workspace_tensors);
            _impl->is_prepared = true;
        }
    }
}
