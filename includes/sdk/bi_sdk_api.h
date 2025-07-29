//
// Created by holynova on 25-4-18.
//

#pragma once

#include <vector>

// This interface is only compiled for the ARM platform.
#if defined(_MSC_VER)
#   if !defined(BIAPI)
#       define BIAPI __stdcall
#   endif
#else
#   ifndef BIAPI
#       define BIAPI __attribute__((visibility("default")))
#   endif // BIAPI
#endif

/**
 * Types of supported models.
 */
enum class BIModelTypes {
    BIGpt2 = 0,
};

/**
 * Return code for interface classes.
 */
enum class BIErrCode {
    BISuccess = 0,
    BIGeneralError,
    BISizeNotMatch,
    BIInvalidArgument,
    BIResDamaged,
    BIResNotExists,
};

#define CHECK_SUCCESS(ret) if (ret != BIErrCode::BISuccess) { return ret; }

/**
 * 模型接口类
 */
class BIModelInterfaceBase {
public:
    virtual ~BIModelInterfaceBase()  = default;

    /**
     * 根据传入的模型参数初始化模型
     * @param data_in 模型的全部参数
     * @param data_size 模型参数的总字节数
     * @param output_vec 初始化时，需要推理一下 </s>，该值对应 </s> 模型输出
     * @param kv_cache_id 模型推理 </s> 时，Attention模块的 kv值会进行记录，该值表示对应 kv cache 的 id值
     * @return 返回码
     */
    virtual BIErrCode bi_init(const char *data_in, size_t data_size, std::vector< std::vector<float> > &output_vec, unsigned int &kv_cache_id) = 0;

    /**
     * 设置模型的输入
     * @param input_vec 模型输入
     * @param kv_cache_id_map 推理时生成的 kv值存储地址
     * @return 返回码
     */
    virtual BIErrCode bi_set_input(std::vector< std::vector<unsigned int> >& input_vec, std::vector< std::vector<unsigned int> > &kv_cache_id_map) = 0;

    /**
     * 调用模型的推理过程
     * @param output_vec 模型输出
     * @param kv_block_ids 模型推理过程中的 kv cache block id 信息
     * @return 返回码
     */
    virtual BIErrCode bi_run(std::vector< std::vector<float> > &output_vec, std::vector<unsigned int> &kv_block_ids, bool is_init) = 0;

    /**
     * kv_cache_block 在推理过程中，有大量无用节点产生，需要在每一步推理前将无用 kv_cache_block 释放掉
     * @param kv_block_ids 需要被释放掉的 kv_block_ids 值 (当前仅传递被释放掉的叶子节点即可，根节点也会随之释放)
     */
    virtual void bi_release_kvcache_block(std::vector<unsigned int> &kv_block_ids) = 0;

    /**
     * kv_cache_block 在推理过程中，可能存在额外向后推理 </s> 的操作，当下一次进行推理时需要将本次 kv_cache_block 释放掉但不能将其路径上所有节点都释放
     * @param kv_block_ids 需要被释放掉的 kv_block_ids 值 (当前仅传递被释放掉的叶子节点，仅叶子节点被释放)
     */
    virtual void bi_release_kvcache_leaf_block(std::vector<unsigned int> &kv_block_ids) = 0;
    
    /**
     * 获取可用的 kv_cache_block 块数量
     * @param avaliable_kvblock_count 记录可用的 kv_cache_block 块数量
     */
    virtual void bi_get_avaliable_kvblock_count(unsigned int &avaliable_kvblock_count) = 0;

    /**
     * "清空" kv_cache，将 kv_cache 指针指向 </s> 推理时的 kv值存储地址
     * @param kv_cache_id </s> 推理时的 kv值存储地址
     * @return 返回码
     */
    virtual BIErrCode bi_reset(unsigned int &kv_cache_id) = 0;

    /**
     * 设置CPU占用核数
     * @param num_threads 占用线程数
     */
    virtual void set_threads_num(unsigned int num_threads) = 0;
};

extern "C" {
    BIModelInterfaceBase* BIAPI CreateBIModelInterface(const BIModelTypes model_type);
    void BIAPI DeleteBIModelInterface(const BIModelInterfaceBase *bi_model_interface);
}

