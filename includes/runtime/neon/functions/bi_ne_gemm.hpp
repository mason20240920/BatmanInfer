//
// Created by Mason on 2025/1/2.
//

#ifndef BATMANINFER_BI_NE_GEMM_HPP
#define BATMANINFER_BI_NE_GEMM_HPP

#include <function_info/bi_GEMMInfo.h>
#include <runtime/bi_i_function.hpp>
#include <runtime/bi_i_memory_manager.hpp>
#include <runtime/bi_i_weights_manager.hpp>
#include <runtime/bi_memory_manager_on_demand.hpp>

namespace BatmanInfer {
    /** 基本函数用于执行 GEMM。此函数调用以下内核：
     *
     * -# cpu::CpuGemm
     * */
    class BINEGEMM : public BIIFunction {
    private:
        struct Impl;
        std::unique_ptr<Impl> _impl;
    };
}

#endif //BATMANINFER_BI_NE_GEMM_HPP
