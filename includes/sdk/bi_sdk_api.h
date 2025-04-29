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
     * @return 返回码
     */
    virtual BIErrCode bi_init(const char *data_in, size_t data_size) = 0;

    /**
     * 设置模型的输入
     * @param input_vec 模型输入
     * @return 返回码
     */
    virtual BIErrCode bi_set_input(std::vector< std::vector<unsigned int> >& input_vec) = 0;

    /**
     * 调用模型的推理过程
     * @param output_vec 模型输出
     * @return 返回码
     */
    virtual BIErrCode bi_run(std::vector< std::vector<float> > &output_vec) = 0;

};

extern "C" {
    BIModelInterfaceBase* BIAPI CreateBIModelInterface(const BIModelTypes model_type);
    void BIAPI DeleteBIModelInterface(const BIModelInterfaceBase *bi_model_interface);
}

