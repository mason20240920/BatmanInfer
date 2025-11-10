//
// Created by holynova on 25-4-18.
//

#include "sdk/bi_sdk_api.h"

#include "model_interface/gpt2_model.h"

int max_seq_len    = 16;
int max_batch_size = 50;
int dict_size      = 6004;
int hidden_size    = 768;
int tensor_max_dim = 6;
int layer_num      = 1;
int head_bs        = 64;

extern "C" {
    BIModelInterfaceBase* BIAPI CreateBIModelInterface(const BIModelTypes model_type) {
        if (BIModelTypes::BIGpt2 == model_type) {
            return new BIGPT2Model(16, 50, 6004, 768, 6, 1, 64);
        } else if (BIModelTypes::BIGpt2_layer3 == model_type) {
            return new BIGPT2Model(16, 50, 6004, 512, 6, 3, 64);
        } else if (BIModelTypes::BIGpt2_layer6 == model_type) {
            return new BIGPT2Model(16, 50, 6004, 512, 6, 6, 64);
        }

        // default return
        return nullptr;
    }

    void BIAPI DeleteBIModelInterface(const BIModelInterfaceBase *bi_model_interface) {
        delete bi_model_interface;
    }
}

