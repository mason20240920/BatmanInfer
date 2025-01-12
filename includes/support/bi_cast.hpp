//
// Created by Mason on 2025/1/12.
//

#ifndef BATMANINFER_BI_CAST_HPP
#define BATMANINFER_BI_CAST_HPP

#include <data/core/bi_error.h>

namespace BatmanInfer {
    namespace utils {
        namespace cast {
            /** Polymorphic cast between two types
 *
 * @warning Will throw an exception if cast cannot take place
 *
 * @tparam Target Target to cast type
 * @tparam Source Source from cast type
 *
 * @param[in] v Value to cast
 *
 * @return The casted value
 */
            template<typename Target, typename Source>
            inline Target polymorphic_cast(Source *v) {
                if (dynamic_cast<Target>(v) == nullptr) {
                    BI_COMPUTE_THROW(std::bad_cast());
                }
                return static_cast<Target>(v);
            }

/** Polymorphic down cast between two types
 *
 * @warning Will assert if cannot take place
 *
 * @tparam Target Target to cast type
 * @tparam Source Source from cast type
 *
 * @param[in] v Value to cast
 *
 * @return The casted value
 */
            template<typename Target, typename Source>
            inline Target polymorphic_downcast(Source *v) {
                BI_COMPUTE_ERROR_ON(dynamic_cast<Target>(v) != static_cast<Target>(v));
                return static_cast<Target>(v);
            }

/** Polymorphic cast between two unique pointer types
 *
 * @warning Will throw an exception if cast cannot take place
 *
 * @tparam Target  Target to cast type
 * @tparam Source  Source from cast type
 * @tparam Deleter Deleter function type
 *
 * @param[in] v Value to cast
 *
 * @return The casted value
 */
            template<typename Target, typename Source, typename Deleter>
            std::unique_ptr<Target, Deleter> polymorphic_cast_unique_ptr(std::unique_ptr<Source, Deleter> &&v) {
                if (dynamic_cast<Target *>(v.get()) == nullptr) {
                    BI_COMPUTE_THROW(std::bad_cast());
                }
                auto r = static_cast<Target *>(v.release());
                return std::unique_ptr<Target, Deleter>(r, std::move(v.get_deleter()));
            }

/** Polymorphic down cast between two unique pointer types
 *
 * @warning Will assert if cannot take place
 *
 * @tparam Target  Target to cast type
 * @tparam Source  Source from cast type
 * @tparam Deleter Deleter function type
 *
 * @param[in] v Value to cast
 *
 * @return The casted value
 */
            template<typename Target, typename Source, typename Deleter>
            std::unique_ptr<Target, Deleter> polymorphic_downcast_unique_ptr(std::unique_ptr<Source, Deleter> &&v) {
                BI_COMPUTE_ERROR_ON(dynamic_cast<Target *>(v.get()) != static_cast<Target *>(v.get()));
                auto r = static_cast<Target *>(v.release());
                return std::unique_ptr<Target, Deleter>(r, std::move(v.get_deleter()));
            }
        }
    }
}

#endif //BATMANINFER_BI_CAST_HPP
