//
// Created by Mason on 2025/1/9.
//

#ifndef BATMANINFER_FLOAT_OPS_HPP
#define BATMANINFER_FLOAT_OPS_HPP

#include <cmath>
#include <cstdint>
#include <cstdlib>

namespace BatmanInfer {
    namespace helpers {
        namespace float_ops {
            union RawFloat {
                /** Constructor
                 *
                 * @param[in] val Floating-point value
                 */
                explicit RawFloat(float val) : f32(val) {
                }

                /** Extract sign of floating point number
                 *
                 * @return Sign of floating point number
                 */
                int32_t sign() const {
                    return i32 >> 31;
                }

                /** Extract exponent of floating point number
                 *
                 * @return Exponent of floating point number
                 */
                int32_t exponent() const {
                    return (i32 >> 23) & 0xFF;
                }

                /** Extract mantissa of floating point number
                 *
                 * @return Mantissa of floating point number
                 */
                int32_t mantissa() const {
                    return i32 & 0x007FFFFF;
                }

                int32_t i32;
                float f32;
            };

            /** Checks if two floating point numbers are equal given an allowed number of ULPs
             *
             * @param[in] a                First number to compare
             * @param[in] b                Second number to compare
             * @param[in] max_allowed_ulps (Optional) Number of allowed ULPs
             *
             * @return True if number is close else false
             */
            inline bool is_equal_ulps(float a, float b, int max_allowed_ulps = 0) {
                RawFloat ra(a);
                RawFloat rb(b);

                // Check ULP distance
                const int ulps = std::abs(ra.i32 - rb.i32);
                return ulps <= max_allowed_ulps;
            }

            /** Checks if the input floating point number is 1.0f checking if the difference is within a range defined with epsilon
             *
             * @param[in] a       Input floating point number
             * @param[in] epsilon (Optional) Epsilon used to define the error bounds
             *
             * @return True if number is close to 1.0f
             */
            inline bool is_one(float a, float epsilon = 0.00001f) {
                return std::abs(1.0f - a) <= epsilon;
            }

            /** Checks if the input floating point number is 0.0f checking if the difference is within a range defined with epsilon
             *
             * @param[in] a       Input floating point number
             * @param[in] epsilon (Optional) Epsilon used to define the error bounds
             *
             * @return True if number is close to 0.0f
             */
            inline bool is_zero(float a, float epsilon = 0.00001f) {
                return std::abs(0.0f - a) <= epsilon;
            }
        } // namespace float_ops
    } // namespace helpers
} // namespace BatmanInfer

#endif //BATMANINFER_FLOAT_OPS_HPP
