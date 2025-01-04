//
// Created by Mason on 2025/1/4.
//

#ifndef BATMANINFER_BI_SCHEDULER_HPP
#define BATMANINFER_BI_SCHEDULER_HPP

#include <runtime/bi_i_scheduler.hpp>

#include <map>
#include <memory>

namespace BatmanInfer {
    /**
     * @brief 可配置的调度器，支持多种多线程 API，并可在运行时选择不同的调度器。
     */
    class BIScheduler {
    public:
        /**
         * @brief 调度类型
         */
        enum class Type {
            ST,    /**< Single thread. */
            CPP,   /**< C++11 threads. */
            OMP,   /**< OpenMP. */
            CUSTOM /**< Provided by the user. */
        };

        /** 设置用户定义的调度器并将其设为活动调度器。
         *
         * @param[in] scheduler 一个由用户实现的自定义调度器的共享指针。
         */
        static void set(std::shared_ptr<BIIScheduler> scheduler);

        /** 访问调度器单例。
        *
        * @return 返回对调度器对象的引用。
        */
        static BIIScheduler &get();

        /** 设置活动调度器。
        *
        * 同一时间只能启用一个调度器。
        *
        * @param[in] t 要启用的调度器类型。
        */
        static void set(Type t);

        /** 返回活动调度器的类型。
        *
        * @return 当前调度器的类型。
        */
        static Type get_type();

        /** 如果给定的调度器类型被支持，则返回 true，否则返回 false。
        *
        * @param[in] t 要检查的调度器类型。
        *
        * @return 如果给定的调度器类型被支持，返回 true；否则返回 false。
        */
        static bool is_available(Type t);

    private:
        static Type                                          _scheduler_type;
        static std::shared_ptr<BIIScheduler>                 _custom_scheduler;
        static std::map<Type, std::unique_ptr<BIIScheduler>> _schedulers;

        BIScheduler();
    };
}

#endif //BATMANINFER_BI_SCHEDULER_HPP
