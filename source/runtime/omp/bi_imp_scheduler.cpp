//
// Created by Mason on 2025/1/4.
//

#include <runtime/omp/bi_imp_scheduler.hpp>

#include <data/core/cpp/bi_i_cpp_kernel.hpp>
#include <data/core/bi_error.h>
#include <data/core/bi_helpers.hpp>
#include <data/core/bi_utils.hpp>

#include <omp.h>

#include "kv_cache_manager/block/physical_block.hpp"
#include "model_interface/gpt2_model.h"
#include "runtime/bi_tensor.hpp"

namespace BatmanInfer {
    template<typename T>
    void print_offset(void *data_ptr, const size_t move_size) {
        T *p = static_cast<T *>(data_ptr);
        for (size_t i = 0; i < move_size; i++) {
            std::cout << static_cast<float>(p[i]) << ", ";
        }
        std::cout << std::endl;
    }
#if !defined(_WIN64) && !defined(BARE_METAL) && !defined(__APPLE__) && !defined(__OpenBSD__) && \
    (defined(__arm__) || defined(__aarch64__)) && defined(__ANDROID__)
    BIOMPScheduler::BIOMPScheduler() :  _num_threads(cpu_info().get_cpu_num_excluding_little()),
                                        _nonlittle_num_cpus(cpu_info().get_cpu_num_excluding_little()) {}
#else

    BIOMPScheduler::BIOMPScheduler() : _num_threads(omp_get_max_threads()),
                                       _nonlittle_num_cpus(cpu_info().get_cpu_num_excluding_little()) {
    }

#endif

    unsigned int BIOMPScheduler::num_threads() const {
        return _num_threads;
    }

    void BIOMPScheduler::set_num_threads(unsigned int num_threads) {
        const unsigned int num_cores = omp_get_max_threads();
#if !defined(_WIN64) && !defined(BARE_METAL) && !defined(__APPLE__) && !defined(__OpenBSD__) && \
    (defined(__arm__) || defined(__aarch64__)) && defined(__ANDROID__)
        const unsigned int adjusted_num_threads = std::min(_nonlittle_num_cpus, num_threads);
    _num_threads                            = (num_threads == 0) ? num_cores : adjusted_num_threads;
#else  /* !defined(_WIN64) && !defined(BARE_METAL) && !defined(__APPLE__) && !defined(__OpenBSD__) && \
    (defined(__arm__) || defined(__aarch64__)) && defined(__ANDROID__)*/
        _num_threads = (num_threads == 0) ? num_cores : num_threads;
#endif /* !defined(_WIN64) && !defined(BARE_METAL) && !defined(__APPLE__) && !defined(__OpenBSD__) && \
    (defined(__arm__) || defined(__aarch64__)) && defined(__ANDROID__)*/
    }

    void BIOMPScheduler::schedule(BatmanInfer::BIICPPKernel *kernel, const BatmanInfer::BIIScheduler::Hints &hints) {
        BIITensorPack tensors;
        schedule_common(kernel, hints, kernel->window(), tensors);
    }

    void BIOMPScheduler::schedule_op(BatmanInfer::BIICPPKernel *kernel, const BatmanInfer::BIIScheduler::Hints &hints,
                                     const BatmanInfer::BIWindow &window, BatmanInfer::BIITensorPack &tensors) {
        // The rest of the logic in this function does not handle the
        // split_dimensions_all case so we defer to IScheduler::schedule_common()
        if (hints.split_dimension() == BIIScheduler::split_dimensions_all) {
            return schedule_common(kernel, hints, window, tensors);
        }

        BI_COMPUTE_ERROR_ON_MSG(!kernel, "The child class didn't set the kernel");
        BI_COMPUTE_ERROR_ON_MSG(hints.strategy() == BIStrategyHint::DYNAMIC,
                                "Dynamic scheduling is not supported in OMPScheduler");

        const BIWindow &max_window = window;
        const unsigned int num_iterations = max_window.num_iterations(hints.split_dimension());
        const unsigned int mws = kernel->get_mws(CPUInfo::get(), _num_threads);

        // Ensure each thread has mws amount of work to do (i.e. ceil(num_iterations / mws) threads)
        const unsigned int candidate_num_threads = (num_iterations + mws - 1) / mws;

        // Cap the number of threads to be spawn with the size of the thread pool
        const unsigned int num_threads = std::min(candidate_num_threads, _num_threads);

        if (!kernel->is_parallelisable() || num_threads == 1) {
            ThreadInfo info;
            info.cpu_info = &cpu_info();
            kernel->run_op(tensors, max_window, info);
        } else {
            const unsigned int num_windows = num_threads;
            std::vector<BIIScheduler::BIWorkload> workloads(num_windows);
            for (unsigned int t = 0; t < num_windows; t++) {
                //Capture 't' by copy, all the other variables by reference:
                workloads[t] = [t, &hints, &max_window, &num_windows, &kernel, &tensors](const ThreadInfo &info) {
                    BIWindow win = max_window.split_window(hints.split_dimension(), t, num_windows);
                    win.validate();
                    kernel->run_op(tensors, win, info);
                };
            }
            run_workloads(workloads);
        }
    }

    void BIOMPScheduler::schedule_kv_split(BIITensorPack &tensors, const std::vector<size_t> &ava_len) {
        // 最后一个维度的数量
        BI_COMPUTE_ERROR_ON_MSG(tensors.empty(), "Tensors are empty in this scheduler!");
        BI_COMPUTE_ERROR_ON_MSG(tensors.get_tensor(ACL_SRC) == nullptr,
                                "Tensors are empty in this scheduler!");
        // 目前使用一个tensor(默认id为0)
        auto *tensor = reinterpret_cast<BITensor *>(tensors.get_tensor(ACL_SRC));
        auto *dst = reinterpret_cast<BITensor *>(tensors.get_tensor(ACL_DST));
        const unsigned int batch = tensor->info()->dimension(2); // batch
        const unsigned int max_seq_len = tensor->info()->dimension(1);
        const unsigned int hidden_size = tensor->info()->dimension(0);
        // 序列长度
        const unsigned int num_iterations = batch;

        // Cap the number of threads to be spawn with the size of the thread pool
        const unsigned int num_threads = std::min(num_iterations, _num_threads);

        // 进行多线程拷贝操作
        const unsigned int num_windows = num_threads;
        const unsigned int total_usage = num_iterations / num_threads; // 每个线程需要处理的维度
        const unsigned int remain_usage = num_iterations % num_threads;
        std::vector<BIIScheduler::BIWorkload> workloads(num_windows);
        for (unsigned int t = 0; t < num_windows; t++) {
            const unsigned int data_type = sizeof(float16_t);
            workloads[t] = [t, &tensor,&dst, &num_windows, &remain_usage, & total_usage,&data_type, &hidden_size, &ava_len, &max_seq_len](const ThreadInfo &info) {
                        // 当前全局的起始节点
                        for (int split_i = 0; split_i < total_usage; split_i++) {
                            unsigned int current_gmem_index = total_usage * t + split_i;
                            // 获取当前坐标
                            // TODO: 当前默认是float16, 所以最后乘以2
                            unsigned int offset = (get_vec_sum(ava_len, current_gmem_index, max_seq_len) - 1)*hidden_size * data_type;
                            unsigned int dst_offset = current_gmem_index * hidden_size * data_type;
                            unsigned int data_size = hidden_size * data_type;
                            auto data = tensor->allocator()->data(offset, data_size);
                            auto dst_ptr = dst->allocator()->data(dst_offset, data_size);
                            memcpy(dst_ptr, data, data_size);
                        }
                        // 计算需要Split开始的起点
                        if (t == num_windows - 1 && remain_usage > 0) {
                            for (int split_i = 0; split_i < remain_usage; split_i++) {
                                unsigned int current_gmem_index = total_usage * (t + 1) + split_i;
                                // 获取当前坐标
                                // TODO: 当前默认是float16, 所以最后乘以2
                                unsigned int offset = (get_vec_sum(ava_len, current_gmem_index, max_seq_len) - 1) * hidden_size * data_type;
                                unsigned int dst_offset = current_gmem_index * hidden_size * data_type;
                                unsigned int data_size = hidden_size * data_type;
                                auto data = tensor->allocator()->data(offset, data_size);
                                auto dst_ptr = dst->allocator()->data(dst_offset, data_size);
                                memcpy(dst_ptr, data, data_size);
                            }
                        }
                    };
        }
        // 进行并发执行
        run_workloads(workloads);
    }

    void BIOMPScheduler::schedule_kv_full_fill(BIITensorPack &tensors, const std::vector<PhysicalBlock *> &mem_lst, const std::vector<size_t> &ava_len) {
        // 1. 最后一个维度的数量
        BI_COMPUTE_ERROR_ON_MSG(tensors.empty(), "Tensors are empty in this scheduler!");
        BI_COMPUTE_ERROR_ON_MSG(tensors.get_tensor(ACL_SRC) == nullptr, "Tensors are empty in this scheduler!");
        BI_COMPUTE_ERROR_ON_MSG(tensors.get_tensor(ACL_DST) == nullptr, "Tensors are empty in this scheduler!");
        auto *k_dst = reinterpret_cast<BITensor *>(tensors.get_tensor(ACL_DST_0));
        auto *v_dst = reinterpret_cast<BITensor *>(tensors.get_tensor(ACL_DST_1));
        const auto k_type_size = k_dst->info()->data_type() == BIDataType::F16 ? sizeof(float16_t) : sizeof(int8_t);
        const auto v_type_size = v_dst->info()->data_type() == BIDataType::F16 ? sizeof(float16_t) : sizeof(int8_t);
        // 2. 先确定当前的并行数量
        const size_t batch = v_dst->info()->dimension(3);
        const size_t max_seq_len = v_dst->info()->dimension(2);
        const size_t num_head = v_dst->info()->dimension(1);
        const size_t dim_size = v_dst->info()->dimension(0);
        const size_t k_mm_dim_item_size = dim_size * k_type_size * num_head;
        const size_t v_mm_dim_item_size = dim_size * v_type_size * num_head;
        const size_t parallel_iter = batch; // 按照batch进行切割
        // 3. 获取当前并行数量
        const size_t num_threads = std::min(parallel_iter, static_cast<size_t>(_num_threads));
        // 4. round up 这些参数
        const size_t middle_loop_num = parallel_iter / num_threads;
        const size_t remain_loop_num = parallel_iter % num_threads;
        // 5. 建立并行匿名函数
        std::vector<BIWorkload> workloads(num_threads);
        // 一般来说是10个线程
        for (unsigned int t = 0; t < num_threads; t++) {
            workloads[t] = [t,
                        &middle_loop_num,
                        &ava_len,
                        &max_seq_len,
                        &k_dst,
                        &v_dst,
                        &k_mm_dim_item_size,
                        &v_mm_dim_item_size,
                        &remain_loop_num,
                        &num_threads,
                        &mem_lst](const ThreadInfo &info) {
                        // 1. 先直接进行中循环遍历
                        for (int m_r = 0; m_r < middle_loop_num; m_r++) {
                            // 2. 对于全局张量的当前偏移量(每个线程偏移middle_loop_num的数量)
                            const unsigned int batch_index = middle_loop_num * t + m_r;
                            auto seq_len = ava_len[batch_index];
                            for (size_t seq_index = seq_len; seq_index < max_seq_len; seq_index++) {
                                const unsigned int cur_k_mm_offset = (batch_index * max_seq_len + seq_index) * k_mm_dim_item_size;
                                const unsigned int cur_v_mm_offset = (batch_index * max_seq_len + seq_index) * v_mm_dim_item_size;
                                auto pb_gmem_addr = static_cast<char *>(mem_lst[seq_index]->buffer);
                                auto k_dst_ptr = k_dst->allocator()->data(cur_k_mm_offset, k_mm_dim_item_size);
                                auto v_dst_ptr = v_dst->allocator()->data(cur_v_mm_offset, v_mm_dim_item_size);
                                memcpy(k_dst_ptr, pb_gmem_addr, k_mm_dim_item_size);
                                memcpy(v_dst_ptr, pb_gmem_addr + k_mm_dim_item_size, v_mm_dim_item_size);
                            }
                        }
                        // 如果是remain大小就进行remain计算
                        if (t == num_threads - 1 && remain_loop_num > 0) {
                            // 1. 先直接进行中循环遍历
                            for (int m_r = 0; m_r <= remain_loop_num; m_r++) {// 2. 对于全局张量的当前偏移量(每个线程偏移middle_loop_num的数量)
                                const unsigned int batch_index = middle_loop_num * t + m_r;
                                auto seq_len = ava_len[batch_index];
                                for (size_t seq = seq_len; seq < max_seq_len; seq++) {
                                    const unsigned int cur_k_mm_offset = (batch_index * max_seq_len + seq) * k_mm_dim_item_size;
                                    const unsigned int cur_v_mm_offset = (batch_index * max_seq_len + seq) * v_mm_dim_item_size;
                                    auto pb_gmem_addr = static_cast<char *>(mem_lst[seq]->buffer);
                                    auto k_dst_ptr = k_dst->allocator()->data(cur_k_mm_offset, k_mm_dim_item_size);
                                    auto v_dst_ptr = v_dst->allocator()->data(cur_v_mm_offset, v_mm_dim_item_size);
                                    memcpy(k_dst_ptr, pb_gmem_addr, k_mm_dim_item_size);
                                    memcpy(v_dst_ptr, pb_gmem_addr + k_mm_dim_item_size, v_mm_dim_item_size);
                                }
                            }
                        }
                    };
        }
        // 执行并行操作
        run_workloads(workloads);
    }

    void BIOMPScheduler::schedule_kv_concat(BIITensorPack &tensors, const std::vector<PhysicalBlock *> &mem_lst,
                                            const std::vector<size_t> &ava_len) {
        // 1. 最后一个维度的数量
        BI_COMPUTE_ERROR_ON_MSG(tensors.empty(), "Tensors are empty in this scheduler!");
        BI_COMPUTE_ERROR_ON_MSG(tensors.get_tensor(ACL_SRC) == nullptr, "Tensors are empty in this scheduler!");
        BI_COMPUTE_ERROR_ON_MSG(tensors.get_tensor(ACL_DST) == nullptr, "Tensors are empty in this scheduler!");
        auto *k_src = reinterpret_cast<BITensor *>(tensors.get_tensor(ACL_SRC_0));
        auto *v_src = reinterpret_cast<BITensor *>(tensors.get_tensor(ACL_SRC_1));
        auto *k_dst = reinterpret_cast<BITensor *>(tensors.get_tensor(ACL_DST_0));
        auto *v_dst = reinterpret_cast<BITensor *>(tensors.get_tensor(ACL_DST_1));
        const auto k_type_size = k_src->info()->data_type() == BIDataType::F16 ? sizeof(float16_t) : sizeof(int8_t);
        const auto v_type_size = v_src->info()->data_type() == BIDataType::F16 ? sizeof(float16_t) : sizeof(int8_t);
        // 2. 先确定当前的并行数量
        const size_t batch = v_dst->info()->dimension(3);
        const size_t max_seq_len = v_dst->info()->dimension(2);
        const size_t num_head = v_dst->info()->dimension(1);
        const size_t dim_size = v_dst->info()->dimension(0);
        const size_t k_mm_dim_item_size = dim_size * k_type_size * num_head;
        const size_t v_mm_dim_item_size = dim_size * v_type_size * num_head;
        const size_t parallel_iter = batch; // 按照batch进行切割
        // 3. 获取当前并行数量
        const size_t num_threads = std::min(parallel_iter, static_cast<size_t>(_num_threads));
        // 4. round up 这些参数
        const size_t middle_loop_num = parallel_iter / num_threads;
        const size_t remain_loop_num = parallel_iter % num_threads;
        // 5. 建立并行匿名函数
        std::vector<BIWorkload> workloads(num_threads);
        // 一般来说是10个线程
        for (unsigned int t = 0; t < num_threads; t++) {
            workloads[t] = [t,
                        &middle_loop_num,
                        &ava_len,
                        &max_seq_len,
                        &k_dst,
                        &v_dst,
                        &k_mm_dim_item_size,
                        &v_mm_dim_item_size,
                        &remain_loop_num,
                        &num_threads,
                        &mem_lst,
                        &k_src,
                        &v_src](const ThreadInfo &info) {
                        // 1. 先直接进行中循环遍历
                        for (int m_r = 0; m_r < middle_loop_num; m_r++) {
                            // 2. 对于全局张量的当前偏移量(每个线程偏移middle_loop_num的数量)
                            const unsigned int batch_index = middle_loop_num * t + m_r;
                            auto seq_len = ava_len[batch_index];
                            for (int seq = 0; seq < seq_len - 1; seq++) {
                                const unsigned int mm_item_index = get_remain_seq_sum_minus_one(ava_len, batch_index) + seq;
                                const unsigned int cur_k_mm_offset = (batch_index * max_seq_len + seq) * k_mm_dim_item_size;
                                const unsigned int cur_v_mm_offset = (batch_index * max_seq_len + seq) * v_mm_dim_item_size;
                                auto pb_gmem_addr = static_cast<char *>(mem_lst[mm_item_index]->buffer);
                                auto k_dst_ptr = k_dst->allocator()->data(cur_k_mm_offset, k_mm_dim_item_size);
                                auto v_dst_ptr = v_dst->allocator()->data(cur_v_mm_offset, v_mm_dim_item_size);
                                memcpy(k_dst_ptr, pb_gmem_addr, k_mm_dim_item_size);
                                memcpy(v_dst_ptr, pb_gmem_addr + k_mm_dim_item_size, v_mm_dim_item_size);
                            }
                            const unsigned int t_k_mm_offset = (batch_index * max_seq_len + seq_len - 1) * k_mm_dim_item_size;
                            const unsigned int t_v_mm_offset = (batch_index * max_seq_len + seq_len - 1) * v_mm_dim_item_size;
                            const unsigned int t_k_orin_offset = batch_index * k_mm_dim_item_size;
                            const unsigned int t_v_orin_offset = batch_index * v_mm_dim_item_size;
                            auto k_gmem_addr = k_src->allocator()->data(t_k_orin_offset, k_mm_dim_item_size);
                            auto v_gmem_addr = v_src->allocator()->data(t_v_orin_offset, v_mm_dim_item_size);
                            auto k_dst_ptr = k_dst->allocator()->data(t_k_mm_offset, k_mm_dim_item_size);
                            auto v_dst_ptr = v_dst->allocator()->data(t_v_mm_offset, v_mm_dim_item_size);
                            memcpy(k_dst_ptr, k_gmem_addr, k_mm_dim_item_size);
                            memcpy(v_dst_ptr, v_gmem_addr, v_mm_dim_item_size);
                        }
                        // 如果是remain大小就进行remain计算
                        if (t == num_threads - 1 && remain_loop_num > 0) {
                            // 1. 先直接进行中循环遍历
                            for (int m_r = 0; m_r <= remain_loop_num; m_r++) {
                                // 2. 对于全局张量的当前偏移量(每个线程偏移middle_loop_num的数量)
                                const unsigned int batch_index = middle_loop_num * t + m_r;
                                auto seq_len = ava_len[batch_index];
                                for (int seq = 0; seq < seq_len - 1; seq++) {
                                    const unsigned int mm_item_index = get_remain_seq_sum_minus_one(
                                                                           ava_len, batch_index) + seq;
                                    const unsigned int cur_k_mm_offset =
                                            (get_remain_seq_sum(ava_len, batch_index) + seq) * k_mm_dim_item_size;
                                    const unsigned int cur_v_mm_offset =
                                            (get_remain_seq_sum(ava_len, batch_index) + seq) * v_mm_dim_item_size;
                                    auto pb_gmem_addr = static_cast<char *>(mem_lst[mm_item_index]->buffer);
                                    auto k_dst_ptr = k_dst->allocator()->data(cur_k_mm_offset, k_mm_dim_item_size);
                                    auto v_dst_ptr = v_dst->allocator()->data(cur_v_mm_offset, v_mm_dim_item_size);
                                    memcpy(k_dst_ptr, pb_gmem_addr, k_mm_dim_item_size);
                                    memcpy(v_dst_ptr, pb_gmem_addr + k_mm_dim_item_size, v_mm_dim_item_size);
                                }
                                const unsigned int t_k_mm_offset = (batch_index  * max_seq_len + seq_len) * k_mm_dim_item_size;
                                const unsigned int t_v_mm_offset = (batch_index  * max_seq_len + seq_len) * v_mm_dim_item_size;
                                const unsigned int t_k_orin_offset = batch_index * k_mm_dim_item_size;
                                const unsigned int t_v_orin_offset = batch_index * v_mm_dim_item_size;
                                auto k_gmem_addr = k_src->allocator()->data(t_k_orin_offset, k_mm_dim_item_size);
                                auto v_gmem_addr = v_src->allocator()->data(t_v_orin_offset, v_mm_dim_item_size);
                                auto k_dst_ptr = k_dst->allocator()->data(t_k_mm_offset, k_mm_dim_item_size);
                                auto v_dst_ptr = v_dst->allocator()->data(t_v_mm_offset, v_mm_dim_item_size);
                                memcpy(k_dst_ptr, k_gmem_addr, k_mm_dim_item_size);
                                memcpy(v_dst_ptr, v_gmem_addr, v_mm_dim_item_size);
                            }
                        }
                    };
        }
        // 执行并行操作
        run_workloads(workloads);
    }


#ifndef DOXYGEN_SKIP_THIS


    void BIOMPScheduler::run_workloads(std::vector<BIWorkload> &workloads) {
        // 计算任务的总数，即 workloads 的大小，并将其存储在 amount_of_work 中
        const auto amount_of_work = static_cast<unsigned int>(workloads.size());
        // 线程数不能超过任务数，因此取两者的最小值
        const unsigned int num_threads_to_use = std::min(_num_threads, amount_of_work);

        if (num_threads_to_use < 1)
            return;

        // 存储线程相关的信息
        ThreadInfo info;
        // 当前 CPU 信息的指针
        info.cpu_info = &cpu_info();
        // 实际使用的线程数
        info.num_threads = static_cast<int>(num_threads_to_use);

        // 在非 Android 环境中，omp_num_threads 被设置为 _num_threads，即系统支持的最大线程数
#if !defined(__ANDROID__)
        // Use fixed number of omp threads in the thread pool because changing this
        // in-between kernel execution negatively affects the scheduler performance,
        // possibly switching between X and Y number of threads, causing reconfiguration
        // of the synchronization mechanism. This has been only tested in a subset of
        // operating systems, thus we limit the change using guards.
        const unsigned int omp_num_threads = _num_threads;
#else  /* !__ANDROID__ */
        const unsigned int omp_num_threads = num_threads_to_use;
#endif /* __ANDROID__ */

        /**
         * @brief omp parallel for：并行化 for 循环。
         * firstprivate(info)：每个线程都会复制一份 info 对象（每个线程有独立的副本）。
         * num_threads(omp_num_threads)：指定 OpenMP 使用的线程数。
         * default(shared)：默认情况下，所有变量在线程之间共享（除非显式指定为私有）。
         * proc_bind(close)：绑定线程到接近的处理器核，以减少线程迁移的开销。
         * schedule(static, 1)：静态调度，每个线程分配固定数量的迭代（这里每次分配 1 个任务）
         */
#pragma omp parallel for firstprivate(info) num_threads(omp_num_threads) default(shared) proc_bind(close) \
    schedule(static, 1)
        // 遍历所有任务的索引 wid，范围是 [0, amount_of_work)
        for (unsigned int wid = 0; wid < amount_of_work; ++wid) {
            // 获取当前线程的 ID（tid），这是 OpenMP 提供的函数
            const int tid = omp_get_thread_num();

            // 当前线程的 ID 存储到 info.thread_id 中
            info.thread_id = tid;
            // 执行工作负载
            workloads[wid](info);
        }
    }

#endif
}
