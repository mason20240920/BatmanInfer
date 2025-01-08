//
// Created by Mason on 2025/1/4.
//

#include <runtime/bi_scheduler.hpp>

#include <data/core/bi_error.h>

#if BI_COMPUTE_CPP_SCHEDULER
#include <runtime/cpp/bi_cpp_scheduler.hpp>
#endif /* BI_COMPUTE_CPP_SCHEDULER */

#include <runtime/bi_single_thread_scheduler.hpp>

#if BI_COMPUTE_OPENMP_SCHEDULER

#include <runtime/omp/bi_imp_scheduler.hpp>

#endif /* BI_COMPUTE_OPENMP_SCHEDULER */

using namespace BatmanInfer;

#if !BI_COMPUTE_CPP_SCHEDULER && BI_COMPUTE_OPENMP_SCHEDULER
BIScheduler::Type BIScheduler::_scheduler_type = BIScheduler::Type::OMP;
#elif BI_COMPUTE_CPP_SCHEDULER && !BI_COMPUTE_OPENMP_SCHEDULER
BIScheduler::Type BIScheduler::_scheduler_type = BIScheduler::Type::CPP;
#elif BI_COMPUTE_CPP_SCHEDULER && BI_COMPUTE_OPENMP_SCHEDULER
BIScheduler::Type BIScheduler::_scheduler_type = BIScheduler::Type::CPP;
#else  /* BI_COMPUTE_*_SCHEDULER */
BIScheduler::Type BIScheduler::_scheduler_type = BIScheduler::Type::ST;
#endif /* BI_COMPUTE_*_SCHEDULER */

std::shared_ptr<BIIScheduler> BIScheduler::_custom_scheduler = nullptr;

namespace {
    std::map<BIScheduler::Type, std::unique_ptr<BIIScheduler>> init() {
        std::map<BIScheduler::Type, std::unique_ptr<BIIScheduler>> m;
        m[BIScheduler::Type::ST] = std::make_unique<BISingleThreadScheduler>();
#if defined(BI_COMPUTE_CPP_SCHEDULER)
        m[BIScheduler::Type::CPP] = std::make_unique<BICPPScheduler>();
#endif // defined(BI_COMPUTE_CPP_SCHEDULER)
#if defined(BI_COMPUTE_OPENMP_SCHEDULER)
        m[BIScheduler::Type::OMP] = std::make_unique<BIOMPScheduler>();
#endif // defined(BI_COMPUTE_OPENMP_SCHEDULER)

        return m;
    }
}

std::map<BIScheduler::Type, std::unique_ptr<BIIScheduler>> BIScheduler::_schedulers{};

void BIScheduler::set(BatmanInfer::BIScheduler::Type t) {
    BI_COMPUTE_ERROR_ON(!BIScheduler::is_available(t));
    _scheduler_type = t;
}

bool BIScheduler::is_available(BatmanInfer::BIScheduler::Type t) {
    if (t == Type::CUSTOM) {
        return _custom_scheduler != nullptr;
    } else {
        return _schedulers.find(t) != _schedulers.end();
    }
}

BIScheduler::Type BIScheduler::get_type() {
    return _scheduler_type;
}

BIIScheduler &BIScheduler::get() {
    if (_scheduler_type == Type::CUSTOM) {
        if (_custom_scheduler == nullptr) {
            BI_COMPUTE_ERROR("No custom scheduler has been setup. Call set(std::shared_ptr<BIIScheduler> &scheduler) "
                             "before BIScheduler::get()");
        } else {
            return *_custom_scheduler;
        }
    } else {
        if (_schedulers.empty()) {
            _schedulers = init();
        }

        auto it = _schedulers.find(_scheduler_type);
        if (it != _schedulers.end()) {
            return *it->second;
        } else {
            BI_COMPUTE_ERROR("Invalid Scheduler type");
        }
    }
}

void BIScheduler::set(std::shared_ptr<BIIScheduler> scheduler) {
    _custom_scheduler = std::move(scheduler);
    set(Type::CUSTOM);
}