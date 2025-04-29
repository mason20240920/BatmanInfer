//
// Created by holynova on 25-4-18.
//

#include "sdk/bi_sdk_api.h"

#include "model_interface/gpt2_model.h"

extern "C" {
    BIModelInterfaceBase* BIAPI CreateBIModelInterface(const BIModelTypes model_type) {
        if (BIModelTypes::BIGpt2 == model_type) {
            return new BIGPT2Model();
        }

        // default return
        return nullptr;
    }

    void BIAPI DeleteBIModelInterface(const BIModelInterfaceBase *bi_model_interface) {
        delete bi_model_interface;
    }
}

