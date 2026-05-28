# 推理框架支持 FIX（W4A8）量化版本设计文档

## 背景

当前分支（dev_zzhe_kvcache_layer3_decode）的最新提交支持 AWQ 和 Float 版本的多层（1/3/6层）GPT-2 推理。本设计在此基础上额外支持 FIX（W4A8）量化版本。

FIX 版本特点：权重以 4-bit 存储（W4），激活值以 8-bit 计算（A8），推理过程使用量化算子（BINEAttentionLowpLayer + BINEMLPLayer）。

## 设计决策

| 决策点 | 选择 | 原因 |
|--------|------|------|
| 版本切换方式 | 编译时宏 `#ifdef FIX_VER` | 与项目历史一致，无运行时开销 |
| Block 层架构 | 修改现有 BINEGPT2Block | 保持多层架构统一，改动最小 |
| KV Cache 类型 | fix 版本用 int8（K=int8, V=fp16） | 与旧版本行为一致，节省内存 |
| 权重打包 | fix 和 AWQ 共用 int4 压缩逻辑 | W4A8，打包方式相同 |
| 权重加载 | 新增 `load_weight_tensor_unpack_only`（解包不反量化） | 职责清晰，不影响 AWQ 路径 |
| 量化参数来源 | `decode_layer_scales` 资源（JSON，含多层参数） | 提供 BINEAttentionLowpLayer 所需的 scale/zp |

## 三版本对比

```
AWQ:  int4 packed → 解包为 int8 → 反量化为 fp16 → BINEAttentionLayer + BINEFeedForwardLayer
FIX:  int4 packed → 解包为 int8 → 保持 int8    → BINEAttentionLowpLayer + BINEMLPLayer
Float: fp16 权重直接加载                         → BINEAttentionLayer + BINEFeedForwardLayer
```

## 涉及文件及改动

### 1. CMakeLists_zzhe.txt

恢复编译选项：

```cmake
# 指定当前编译版本
option(FLOAT_VER "float version" OFF)
option(FIX_VER "fix version" OFF)
option(AWQ_VER "awq version" ON)

if(FLOAT_VER)
    add_definitions(-DFLOAT_VER)
endif()
if(FIX_VER)
    add_definitions(-DFIX_VER)
endif()
if(AWQ_VER)
    add_definitions(-DAWQ_VER)
endif()
```

### 2. gpt2_model.h

改动点：
- 重命名 `_c_attn_awq_weight_tensors` → `_c_attn_unpacked_weight_tensors`（存储解包后的数据，fix 和 AWQ 共用）
- 重命名 `_c_fc_awq_weight_tensors` → `_c_fc_unpacked_weight_tensors`（存储解包后的数据，fix 和 AWQ 共用）
- fix 版本：`_c_attn_weight_tensors` 直接指向 unpacked tensor（QSYMM8_PER_CHANNEL）
- AWQ 版本：`_c_attn_weight_tensors` 存储反量化后的 fp16 结果
- 新增 `load_weight_tensor_unpack_only` 函数声明
- 添加 decode_layer_scales 相关存储结构

```cpp
// 成员变量（gpt2_model.h）
std::array<BITensor, 6> _c_attn_weight_tensors;          // fix: QSYMM8, awq/float: F16
std::array<BITensor, 6> _c_attn_unpacked_weight_tensors; // 解包后的数据（fix和awq共用）
std::array<BITensor, 6> _c_fc_weight_tensors;            // fix: QSYMM8, awq/float: F16
std::array<BITensor, 6> _c_fc_unpacked_weight_tensors;   // 解包后的数据（fix和awq共用）

// 新增函数声明
BIErrCode load_weight_tensor_unpack_only(BITensor &tensor, GPT2ResOrder res_order,
    OrderPtrMap &order2ptr, std::vector<float> &scales);
```

### 3. gpt2_model.cpp

改动点：

**构造函数 — KV Cache 初始化：**
```cpp
#ifdef FIX_VER
    // fix版本：K用int8, V用fp16
    KVCacheManager::initialize(2048,
        hidden_size * sizeof(int8_t) + hidden_size * sizeof(float16_t),
        max_seq_len, layer_num);
#else
    // awq/float版本：KV都用fp16
    KVCacheManager::initialize(2048,
        hidden_size * sizeof(float16_t) * 2,
        max_seq_len, layer_num);
#endif
```

**load_all_non_dynamic_tensors — 权重加载：**
```cpp
// fix版本：c_attn_weight 初始化为 QSYMM8_PER_CHANNEL
#ifdef FIX_VER
    _c_attn_weight_tensors[i].allocator()->init(
        BITensorInfo(c_attn_weight_tensor_shape, 1, BIDataType::QSYMM8_PER_CHANNEL));
#else
    _c_attn_weight_tensors[i].allocator()->init(
        BITensorInfo(c_attn_weight_tensor_shape, 1, BIDataType::F16));
#endif

// 加载权重
#ifdef FIX_VER
    // fix: 解包int4→int8，不反量化
    ret = load_weight_tensor_unpack_only(_c_attn_unpacked_weight_tensors[i],
        static_cast<GPT2ResOrder>(...), order2ptr, c_attn_scales);
#elif defined(AWQ_VER)
    // awq: 解包int4→int8→反量化为fp16
    ret = load_weight_tensor_and_dequantization(_c_attn_unpacked_weight_tensors[i],
        _c_attn_weight_tensors[i], ..., c_attn_scales);
#else
    // float: 直接加载fp16
    ret = load_weight_tensors(_c_attn_weight_tensors, ...);
#endif
```

**新增 load_weight_tensor_unpack_only 函数：**
```cpp
BIErrCode BIGPT2Model::load_weight_tensor_unpack_only(
    BITensor &tensor, GPT2ResOrder res_order,
    OrderPtrMap &order2ptr, std::vector<float> &scales) {
    // 1. 读取 int4 packed 数据
    // 2. 调用 unpack_int8_to_int4 解包为 int8
    // 3. 存入 tensor（QSYMM8_PER_CHANNEL）
    // 4. 设置 scale 信息：tensor.info()->set_quantization_info(scales)
    // 不执行反量化
}
```

**加载 decode_layer_scales：**
```cpp
#ifdef FIX_VER
    // 从 decode_layer_scales 资源加载每层的量化参数
    // 包含：gemm_i_scale, gemm_i_zp, attn_gemm_o_scale, attn_gemm_o_zp,
    //       query_q_scale, query_q_zp, value_q_scale, value_q_zp,
    //       key_q_scale, key_q_zp, softmax_out_scale, softmax_out_zp,
    //       pv_bmm_out_scale, pv_bmm_out_zp,
    //       fc1_input_scale, fc1_input_zp, fc1_output_scale, fc1_output_zp,
    //       gelu_output_scale, gelu_output_zp
    // 解析后存入 gpt_layer_configs 的量化参数字段
#endif
```

### 4. BINEMultiGPTBlock.hpp — BIGPTLayerConfig 扩展

```cpp
struct BIGPTLayerConfig {
    // 现有字段保持不变...
    const BIITensor *ln_1_weight;
    const BIITensor *c_attn_weights;
    const BIITensor *c_attn_bias;
    const BIITensor *o_attn_weights;
    const BIITensor *o_attn_bias;
    const BIITensor *fc_weights;
    const BIITensor *fc_bias;
    const BIITensor *proj_weights;
    const BIITensor *proj_bias;
    const BIITensor *ln_2_weight;
    BIActivationLayerInfo act_info;
    int layer_idx;

#ifdef FIX_VER
    // Attention 量化参数
    float gemm_i_scale = 0.f;
    int gemm_i_zp = 0;
    float attn_gemm_o_scale = 0.f;
    int attn_gemm_o_zp = 0;
    float query_q_scale = 0.f;
    int query_q_zp = 0;
    float value_q_scale = 0.f;
    int value_q_zp = 0;
    float key_q_scale = 0.f;
    int key_q_zp = 0;
    float softmax_out_scale = 0.f;
    int softmax_out_zp = 0;
    float pv_bmm_out_scale = 0.f;
    int pv_bmm_out_zp = 0;

    // MLP 量化参数
    float fc1_input_scale = 0.f;
    int fc1_input_zp = 0;
    float fc1_output_scale = 0.f;
    int fc1_output_zp = 0;
    float gelu_output_scale = 0.f;
    int gelu_output_zp = 0;

    // fc_weights 的量化信息指针
    const BIQuantizationInfo *c_fc_weight_qinfo = nullptr;
#endif
};
```

### 5. BINEGPT2Block.hpp

成员变量切换：

```cpp
private:
#ifdef FIX_VER
    BINEAttentionLowpLayer _attn_lowp_layer;  // 量化注意力层
    BINEMLPLayer           _mlp_layer;         // 量化MLP层
#else
    BINEAttentionLayer     _attn_layer;        // fp16注意力层
    BINEFeedForwardLayer   _mlp_layer;         // fp16 MLP层
#endif
    BINEArithmeticAddition _add_layer;
    BINEArithmeticAddition _add_2_layer;
    BINECopy _copy_layer;
```

configure 接口保持不变（通过 BIGPTLayerConfig 传入所有参数），内部根据宏选择不同的层配置逻辑。

### 6. BINEGPT2Block.cpp

**configure 方法：**
```cpp
#ifdef FIX_VER
    _attn_lowp_layer.configure(input, ln_1_weight, c_attn_weights, c_attn_bias,
        o_attn_weights, o_attn_bias, eos_weights,
        config.gemm_i_scale, config.gemm_i_zp,
        config.attn_gemm_o_scale, config.attn_gemm_o_zp,
        config.query_q_scale, config.query_q_zp,
        config.value_q_scale, config.value_q_zp,
        config.key_q_scale, config.key_q_zp,
        config.softmax_out_scale, config.softmax_out_zp,
        config.pv_bmm_out_scale, config.pv_bmm_out_zp,
        q_perm, k_perm, qkv_perm,
        hidden_size, max_seq_len, batch_size, layer_idx, &_sub_attn_output);
#else
    _attn_layer.configure(input, ln_1_weight, c_attn_weights, c_attn_bias,
        o_attn_weights, o_attn_bias, eos_weights,
        q_perm, k_perm, qkv_perm,
        hidden_size, max_seq_len, batch_size, layer_idx, &_sub_attn_output);
#endif
```

**run 方法：**
```cpp
#ifdef FIX_VER
    _attn_lowp_layer.run();
    _attn_lowp_layer.get_kv_block_ids(kv_block_ids);
#else
    _attn_layer.run();
    _attn_layer.get_kv_block_ids(kv_block_ids);
#endif
```

### 7. GPT2ResOrder 枚举

在 `mlp_after_rms_gamma` 之后添加 fix 版本专用资源：

```cpp
    mlp_after_rms_gamma,

#ifdef FIX_VER
    decode_layer_scales,  // JSON格式，包含所有层的量化参数
#endif

    all_res_count,
};
```

### 8. test/test_pack_res.cpp

fix 版本打包改动：
- c_attn_weights 和 c_fc_weights 走 `read_and_write_npy_int8toint4`（与 AWQ 共用）
- c_attn_scales 和 c_fc_scales 走 `read_and_write_scales`（与 AWQ 共用）
- 新增 decode_layer_scales 走 `read_and_write_json`

```cpp
#if defined(FIX_VER) || defined(AWQ_VER)
    case GPT2ResOrder::c_attn_weights_0:
    case GPT2ResOrder::reordered_c_fc_weights_0:
    // ... 其他层
        ret = res_pack::read_and_write_npy_int8toint4(...);
        break;

    case GPT2ResOrder::c_attn_scales_0:
    case GPT2ResOrder::c_fc_scales_0:
    // ... 其他层
        ret = res_pack::read_and_write_scales(...);
        break;
#endif

#ifdef FIX_VER
    case GPT2ResOrder::decode_layer_scales:
        ret = res_pack::read_and_write_json(...);
        break;
#endif
```

## decode_layer_scales JSON 格式

每层包含以下量化参数（多层时为数组）：

```json
{
  "layers": [
    {
      "gemm_i_scale": 0.05,
      "gemm_i_zp": 0,
      "attn_gemm_o_scale": 0.03,
      "attn_gemm_o_zp": 0,
      "query_q_scale": 0.04,
      "query_q_zp": 0,
      "value_q_scale": 0.04,
      "value_q_zp": 0,
      "key_q_scale": 0.04,
      "key_q_zp": 0,
      "softmax_out_scale": 0.007,
      "softmax_out_zp": 0,
      "pv_bmm_out_scale": 0.02,
      "pv_bmm_out_zp": 0,
      "fc1_input_scale": 0.05,
      "fc1_input_zp": 0,
      "fc1_output_scale": 0.03,
      "fc1_output_zp": 0,
      "gelu_output_scale": 0.02,
      "gelu_output_zp": 0
    }
  ]
}
```

## 实现顺序

1. CMakeLists_zzhe.txt — 恢复编译选项
2. GPT2ResOrder 枚举 — 添加 decode_layer_scales
3. BIGPTLayerConfig — 添加量化参数字段
4. BINEGPT2Block.hpp/.cpp — #ifdef 切换层类型和逻辑
5. gpt2_model.h — 添加 load_weight_tensor_unpack_only 声明
6. gpt2_model.cpp — 实现 fix 版本的权重加载和初始化
7. test/test_pack_res.cpp — fix 版本打包支持

## 风险和注意事项

- GPT2ResOrder 枚举末尾添加 `decode_layer_scales` 不影响现有 AWQ/Float 资源的枚举值
- BINEGPT2Block 的公共接口（configure 签名）需要适配新的量化参数传递方式
- KV Cache Manager 的 int8 模式需要确认当前多层版本是否已支持
- 编译时只能选择一个版本（FLOAT_VER/FIX_VER/AWQ_VER 三选一）
