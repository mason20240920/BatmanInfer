//
// Created by Mason on 2024/12/26.
//

#include <runtime/bi_memory.hpp>

namespace BatmanInfer {
    BIMemory::BIMemory() : _region(nullptr), _region_owned(nullptr) {

    }

    BIMemory::BIMemory(const std::shared_ptr<BIIMemoryRegion> &memory) : _region(nullptr), _region_owned(memory) {
        _region_owned = memory;
        _region       = _region_owned.get();
    }

    BIMemory::BIMemory(BatmanInfer::BIIMemoryRegion *memory) : _region(memory), _region_owned(nullptr) {
        _region = memory;
    }

    BIIMemoryRegion *BIMemory::region() {
        return _region;
    }

    BIIMemoryRegion *BIMemory::region() const {
        return _region;
    }

    void BIMemory::set_region(BatmanInfer::BIIMemoryRegion *region) {
        _region_owned = nullptr;
        _region       = region;
    }

    void BIMemory::set_owned_region(std::unique_ptr<BIIMemoryRegion> region) {
        _region_owned = std::move(region);
        _region       = _region_owned.get();
    }
}