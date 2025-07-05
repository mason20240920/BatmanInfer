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
     * @return 返回码
     */
    virtual BIErrCode bi_set_input(std::vector< std::vector<unsigned int> >& input_vec, std::vector< std::vector<unsigned int> > &kv_cache_id_map) = 0;

    /**
     * 调用模型的推理过程
     * @param output_vec 模型输出
     * @param kv_block_ids 模型推理过程中的 kv cache block id 信息
     * @return 返回码
     */
    virtual BIErrCode bi_run(std::vector< std::vector<float> > &output_vec, std::vector<unsigned int> &kv_block_ids) = 0;

};

extern "C" {
    BIModelInterfaceBase* BIAPI CreateBIModelInterface(const BIModelTypes model_type);
    void BIAPI DeleteBIModelInterface(const BIModelInterfaceBase *bi_model_interface);
}

