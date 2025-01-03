//
// Created by Mason on 2025/1/3.
//

#include <data/core/bi_i_tensor_pack.hpp>
#include <data/core/bi_i_tensor.hpp>

namespace BatmanInfer {
    BIITensorPack::BIITensorPack(std::initializer_list<PackElement> l) : _pack(){
        for (auto &e : l) {
            _pack[e.id] = e;
        }
    }

    void BIITensorPack::add_tensor(int id, BatmanInfer::BIITensor *tensor) {
        _pack[id] = PackElement(id, tensor);
    }

    void BIITensorPack::add_tensor(int id, const BatmanInfer::BIITensor *tensor) {
        _pack[id] = PackElement(id, tensor);
    }

    void BIITensorPack::add_const_tensor(int id, const BatmanInfer::BIITensor *tensor) {
        add_tensor(id, tensor);
    }

    const BIITensor *BIITensorPack::get_const_tensor(int id) const {
        auto it = _pack.find(id);
        if (it != _pack.end())
            return it->second.ctensor != nullptr ? it->second.ctensor : it->second.tensor;
        return nullptr;
    }

    BIITensor *BIITensorPack::get_tensor(int id) {
        auto it = _pack.find(id);
        return it != _pack.end() ? it->second.tensor : nullptr;
    }

    void BIITensorPack::remove_tensor(int id) {
        _pack.erase(id);
    }

    size_t BIITensorPack::size() const {
        return _pack.size();
    }

    bool BIITensorPack::empty() const {
        return _pack.empty();
    }
}