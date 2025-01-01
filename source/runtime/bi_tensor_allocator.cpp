//
// Created by Mason on 2024/12/26.
//

#include <data/core/bi_tensor_info.hpp>
#include <runtime/bi_tensor_allocator.hpp>
#include <runtime/bi_memory_region.hpp>

#include <data/core/utils/misc/utils.hpp>

using namespace BatmanInfer;

namespace {
    /**
     * @brief 验证子张量形状是否与父张量形状兼容。
     * @param parent_info
     * @param child_info
     * @param coords
     * @return
     */
    bool validate_sub_tensor_shape(const BITensorInfo &parent_info,
                                   const BITensorInfo &child_info,
                                   const BICoordinates &coords) {
        // 初始化验证结果为 true，假设子张量形状是有效的
        bool is_valid = true;

        // 获取父张量和子张量的形状
        const BITensorShape &parent_shape = parent_info.tensor_shape();
        const BITensorShape &child_shape  = child_info.tensor_shape();

        // 获取父张量和子张量的维度数
        const size_t parent_dims = parent_info.num_dimensions();
        const size_t child_dims  = child_info.num_dimensions();

        // 如果子张量的维度数小于或等于父张量的维度数
        if (child_dims <= parent_dims) {
            // 从子张量的最高维度（最后一个维度）开始逐个验证
            for (size_t num_dimensions = child_dims; num_dimensions > 0; --num_dimensions) {
                // 计算子张量在当前维度的结束位置
                const size_t child_dim_size = coords[num_dimensions - 1] + child_shape[num_dimensions - 1];

                // 检查以下条件：
                // 1. 子张量的起始坐标不能为负值。
                // 2. 子张量的结束位置不能超过父张量对应维度的大小。
                if ((coords[num_dimensions - 1] < 0) || (child_dim_size > parent_shape[num_dimensions - 1])) {
                    is_valid = false;
                    break;
                }
            }
        } else
            // 如果子张量的维度数大于父张量的维度数，直接认为无效
            is_valid = false;

        return is_valid;
    }
}

BITensorAllocator::BITensorAllocator(BIIMemoryManageable *owner) : _owner(owner), _associated_memory_group(nullptr),
                                                                   _memory() {

}

BITensorAllocator::~BITensorAllocator() {
    info().set_is_resizable(true);
}

BITensorAllocator::BITensorAllocator(BITensorAllocator &&o) noexcept: BIITensorAllocator(std::move(o)),
                                                                      _owner(o._owner),
                                                                      _associated_memory_group(
                                                                              o._associated_memory_group),
                                                                      _memory(std::move(o._memory)) {
    // 转移之后把所有的值进行清空
    o._owner                   = nullptr;
    o._associated_memory_group = nullptr;
    o._memory                  = BIMemory();
}

BITensorAllocator &BITensorAllocator::operator=(BatmanInfer::BITensorAllocator &&o) noexcept {
    if (&o != this) {
        _owner = o._owner;
        o._owner = nullptr;

        _associated_memory_group = o._associated_memory_group;
        o._associated_memory_group = nullptr;

        _memory = std::move(o._memory);
        o._memory = BIMemory();

        BIITensorAllocator::operator=(std::move(o));
    }
    return *this;
}

void BITensorAllocator::init(const BITensorAllocator &allocator,
                             const BICoordinates &coords,
                             BITensorInfo &sub_info) {
    // 获取父分配器的信息
    const BITensorInfo& parent_info = allocator.info();

    // 检查坐标和新形状是否在父张量内。
    BI_COMPUTE_ERROR_ON(!validate_sub_tensor_shape(parent_info, sub_info, coords));
    BI_COMPUTE_UNUSED(validate_sub_tensor_shape);


    // 拷贝指针到缓存
    _memory = BIMemory(allocator._memory.region());

    // 用新维度初始化张量信息
    size_t total_size =  parent_info.offset_element_in_bytes(coords) + sub_info.total_size() - sub_info.offset_first_element_in_bytes();

    sub_info.init(sub_info.tensor_shape(), sub_info.format(), parent_info.strides_in_bytes(),
                  parent_info.offset_element_in_bytes(coords), total_size);

    // 设置张量信息
    init(sub_info);
}

uint8_t *BITensorAllocator::data() const {
    return (_memory.region() == nullptr) ? nullptr : reinterpret_cast<uint8_t *>(_memory.region()->buffer());
}

void BITensorAllocator::allocate() {
    // 如果未指定对齐方式，则默认对齐到64字节边界。
    const size_t alignment_to_use = (alignment() != 0) ? alignment() : 64;
    if (_associated_memory_group == nullptr)
        _memory.set_owned_region(std::make_unique<BIMemoryRegion>(info().total_size(), alignment_to_use));
    else
        _associated_memory_group->finalize_memory(_owner, _memory, info().total_size(), alignment_to_use);
    info().set_is_resizable(false);
}

void BITensorAllocator::free() {
    _memory.set_region(nullptr);
    info().set_is_resizable(true);
}

bool BITensorAllocator::is_allocated() const {
    return _memory.region() != nullptr;
}

BIStatus BITensorAllocator::import_memory(void *memory) {
    BI_COMPUTE_RETURN_ERROR_ON(memory == nullptr);
    BI_COMPUTE_RETURN_ERROR_ON(_associated_memory_group != nullptr);
    BI_COMPUTE_RETURN_ERROR_ON(alignment() != 0 && !misc::utility::check_aligned(memory, alignment()));

    _memory.set_owned_region(std::make_unique<BIMemoryRegion>(memory, info().total_size()));
    info().set_is_resizable(false);

    return BIStatus{};
}

void BITensorAllocator::set_associated_memory_group(BatmanInfer::BIIMemoryGroup *associated_memory_group) {
    BI_COMPUTE_ERROR_ON(associated_memory_group == nullptr);
    BI_COMPUTE_ERROR_ON(_associated_memory_group != nullptr && _associated_memory_group != associated_memory_group);
    BI_COMPUTE_ERROR_ON(_memory.region() != nullptr && _memory.region()->buffer() != nullptr);

    _associated_memory_group = associated_memory_group;
}

uint8_t *BITensorAllocator::lock() {
    BI_COMPUTE_ERROR_ON(_memory.region() == nullptr);
    return reinterpret_cast<uint8_t *>(_memory.region()->buffer());
}

void BITensorAllocator::unlock() {

}