//
// Created by holynova on 2024/12/30.
//

#ifndef BATMANINFER_BI_CORE_ERROR_H
#define BATMANINFER_BI_CORE_ERROR_H

#include <array>
#include <string>

namespace BatmanInfer {

    /** Ignores unused arguments
     *
     * @tparam T Argument types
     *
     * @param[in] ... Ignored arguments
     */
    template <typename... T>
    inline void ignore_unused(T &&...)
    {
    }

    /** Available error codes */
    enum class BIErrorCode
    {
        OK,                       /**< No error */
        RUNTIME_ERROR,            /**< Generic runtime error */
        UNSUPPORTED_EXTENSION_USE /**< Unsupported extension used*/
    };

    /** Status class */
    class BIStatus
    {
    public:
        /** Default Constructor **/
        BIStatus() : _code(BIErrorCode::OK), _error_description(" ")
        {
        }
        /** Default Constructor
         *
         * @param error_status      Error status.
         * @param error_description (Optional) Error description if error_status is not valid.
         */
        explicit BIStatus(BIErrorCode error_status, std::string error_description = " ")
            : _code(error_status), _error_description(error_description)
        {
        }
        /** Allow instances of this class to be copy constructed */
        BIStatus(const BIStatus &) = default;
        /** Allow instances of this class to be move constructed */
        BIStatus(BIStatus &&) = default;
        /** Allow instances of this class to be copy assigned */
        BIStatus &operator=(const BIStatus &) = default;
        /** Allow instances of this class to be move assigned */
        BIStatus &operator=(BIStatus &&) = default;
        /** Explicit bool conversion operator
         *
         * @return True if there is no error else false
         */
        explicit operator bool() const noexcept
        {
            return _code == BIErrorCode::OK;
        }
        /** Gets error code
         *
         * @return Error code.
         */
        BIErrorCode error_code() const
        {
            return _code;
        }
        /** Gets error description if any
         *
         * @return Error description.
         */
        std::string error_description() const
        {
            return _error_description;
        }
        /** Throws a runtime exception in case it contains a valid error status */
        void throw_if_error() const
        {
            if (!bool(*this))
            {
                internal_throw_on_error();
            }
        }

    private:
        /** Internal throwing function */
        [[noreturn]] void internal_throw_on_error() const;

    private:
        BIErrorCode _code;
        std::string _error_description;
    };

    /** Creates an error containing the error message
     *
     * @param[in] error_code Error code
     * @param[in] msg        Message to display before abandoning.
     *
     * @return status containing the error
     */
    BIStatus create_error(BIErrorCode error_code, std::string msg);

    /** Creates an error and the error message
     *
     * @param[in] error_code Error code
     * @param[in] func       Function in which the error occurred.
     * @param[in] file       File in which the error occurred.
     * @param[in] line       Line in which the error occurred.
     * @param[in] msg        Message to display before abandoning.
     *
     * @return status containing the error
     */
    BIStatus create_error_msg(BIErrorCode error_code, const char *func, const char *file, int line, const char *msg);

    /** Throw an std::runtime_error
     *
     * @param[in] err Error status
     */
    [[noreturn]] void throw_error(BIStatus err);

} // namespace BatmanInfer

/** To avoid unused variables warnings
 *
 * This is useful if for example a variable is only used
 * in debug builds and generates a warning in release builds.
 *
 * @param[in] ... Variables which are unused.
 */
#define BI_COMPUTE_UNUSED(...) ::BatmanInfer::ignore_unused(__VA_ARGS__) // NOLINT

/** Creates an error with a given message
 *
 * @param[in] error_code Error code.
 * @param[in] msg        Message to encapsulate.
 */
#define BI_COMPUTE_CREATE_ERROR(error_code, msg) \
    BatmanInfer::create_error_msg(error_code, __func__, __FILE__, __LINE__, msg)

/** Creates an error on location with a given message
 *
 * @param[in] error_code Error code.
 * @param[in] func       Function in which the error occurred.
 * @param[in] file       File in which the error occurred.
 * @param[in] line       Line in which the error occurred.
 * @param[in] msg        Message to display before abandoning.
 */
#define BI_COMPUTE_CREATE_ERROR_LOC(error_code, func, file, line, msg) \
    BatmanInfer::create_error_msg(error_code, func, file, line, msg)

/** Creates an error on location with a given message. Accepts a message format
 *  and a variable list of arguments matching the format description.
 *
 * @param[in] error_code Error code.
 * @param[in] func       Function in which the error occurred.
 * @param[in] file       File in which the error occurred.
 * @param[in] line       Line in which the error occurred.
 * @param[in] msg        Error description message format.
 * @param[in] ...        List of arguments matching the format description.
 */
#define BI_COMPUTE_CREATE_ERROR_LOC_VAR(error_code, func, file, line, msg, ...)                             \
    do                                                                                                      \
    {                                                                                                       \
        std::array<char, 512> out{0};                                                                       \
        int                   offset = snprintf(out.data(), out.size(), "in %s %s:%d: ", func, file, line); \
        snprintf(out.data() + offset, out.size() - offset, msg, __VA_ARGS__);                               \
        BatmanInfer::create_error(error_code, std::string(out.data()));                                     \
    } while (false)

/** An error is returned with the given description.
 *
 * @param[in] ... Error description message.
 */
#define BI_COMPUTE_RETURN_ERROR_MSG(...)                                                     \
    do                                                                                       \
    {                                                                                        \
        return BI_COMPUTE_CREATE_ERROR(arm_compute::ErrorCode::RUNTIME_ERROR, __VA_ARGS__);  \
    } while (false)

/** Checks if a status contains an error and returns it
 *
 * @param[in] status Status value to check
 */
#define BI_COMPUTE_RETURN_ON_ERROR(status)  \
    do                                      \
    {                                       \
        const auto s = status;              \
        if (!bool(s))                       \
        {                                   \
            return s;                       \
        }                                   \
    } while (false)

/** Checks if an error value is valid if not throws an exception with the error
 *
 * @param[in] error Error value to check.
 */
#define BI_COMPUTE_THROW_ON_ERROR(error) error.throw_if_error();

/** If the condition is true, an error is returned. Accepts a message format
 *  and a variable list of arguments matching the format description.
 *
 * @param[in] cond Condition to evaluate.
 * @param[in] msg  Error description message format.
 * @param[in] ...  List of arguments matching the format description.
 */
#define BI_COMPUTE_RETURN_ERROR_ON_MSG_VAR(cond, msg, ...)                                                      \
    do                                                                                                          \
    {                                                                                                           \
        if (cond)                                                                                               \
        {                                                                                                       \
            std::array<char, 512> out{0};                                                                       \
            int offset = snprintf(out.data(), out.size(), "in %s %s:%d: ", __func__, __FILE__, __LINE__);       \
            snprintf(out.data() + offset, out.size() - offset, msg, __VA_ARGS__);                               \
            return BatmanInfer::create_error(BatmanInfer::BIErrorCode::RUNTIME_ERROR, std::string(out.data())); \
        }                                                                                                       \
    } while (false)

/** If the condition is true, an error is returned
*
* @param[in] cond Condition to evaluate.
* @param[in] msg  Error description message
*/
#define BI_COMPUTE_RETURN_ERROR_ON_MSG(cond, msg)                                                             \
    do                                                                                                        \
    {                                                                                                         \
        if (cond)                                                                                             \
        {                                                                                                     \
            return BatmanInfer::create_error_msg(BatmanInfer::BIErrorCode::RUNTIME_ERROR, __func__, __FILE__, \
                                                 __LINE__, msg);                                              \
        }                                                                                                     \
    } while (false)

/** If the condition is true, an error is thrown. Accepts a message format
 *  and a variable list of arguments matching the format description.
 *
 * @param[in] cond Condition to evaluate.
 * @param[in] func Function in which the error occurred.
 * @param[in] file File in which the error occurred.
 * @param[in] line Line in which the error occurred.
 * @param[in] msg  Error description message format.
 * @param[in] ...  List of arguments matching the format description.
 */
#define BI_COMPUTE_RETURN_ERROR_ON_LOC_MSG_VAR(cond, func, file, line, msg, ...)                                \
    do                                                                                                          \
    {                                                                                                           \
        if (cond)                                                                                               \
        {                                                                                                       \
            std::array<char, 512> out{0};                                                                       \
            int                   offset = snprintf(out.data(), out.size(), "in %s %s:%d: ", func, file, line); \
            snprintf(out.data() + offset, out.size() - offset, msg, __VA_ARGS__);                               \
            return BatmanInfer::create_error(BatmanInfer::BIErrorCode::RUNTIME_ERROR, std::string(out.data())); \
        }                                                                                                       \
    } while (false)

/** If the condition is true, an error is thrown.
*
* @param[in] cond Condition to evaluate.
* @param[in] func Function in which the error occurred.
* @param[in] file File in which the error occurred.
* @param[in] line Line in which the error occurred.
* @param[in] msg  Message to display.
*/
#define BI_COMPUTE_RETURN_ERROR_ON_LOC_MSG(cond, func, file, line, msg)                                           \
    do                                                                                                            \
    {                                                                                                             \
        if (cond)                                                                                                 \
        {                                                                                                         \
            return BatmanInfer::create_error_msg(BatmanInfer::BIErrorCode::RUNTIME_ERROR, func, file, line, msg); \
        }                                                                                                         \
    } while (false)

/** If the condition is true, an error is returned
 *
 * @param[in] cond Condition to evaluate
 */
#define BI_COMPUTE_RETURN_ERROR_ON(cond) BI_COMPUTE_RETURN_ERROR_ON_MSG(cond, #cond)

/** If the condition is true, an error is returned
 *
 * @param[in] cond Condition to evaluate.
 * @param[in] func Function in which the error occurred.
 * @param[in] file File in which the error occurred.
 * @param[in] line Line in which the error occurred.
 */
#define BI_COMPUTE_RETURN_ERROR_ON_LOC(cond, func, file, line) \
    BI_COMPUTE_RETURN_ERROR_ON_LOC_MSG(cond, func, file, line, #cond)

/** Print the given message then throw an std::runtime_error.
 *
 * @param[in] func Function in which the error occurred.
 * @param[in] file File in which the error occurred.
 * @param[in] line Line in which the error occurred.
 * @param[in] msg  Message to display.
 */
#define BI_COMPUTE_THROW_ERROR(func, file, line, msg)                                                       \
    do                                                                                                      \
    {                                                                                                       \
        BatmanInfer::throw_error(                                                                           \
            BatmanInfer::create_error_msg(BatmanInfer::BIErrorCode::RUNTIME_ERROR, func, file, line, msg)); \
    } while (false)

/** Print the given message then throw an std::runtime_error. Accepts a message format
*  and a variable list of arguments matching the format description.
*
* @param[in] func Function in which the error occurred.
* @param[in] file File in which the error occurred.
* @param[in] line Line in which the error occurred.
* @param[in] msg  Error description message format.
* @param[in] ...  List of arguments matching the format description.
*/
#define BI_COMPUTE_THROW_ERROR_VAR(func, file, line, msg, ...)                                              \
    do                                                                                                      \
    {                                                                                                       \
        std::array<char, 512> out{0};                                                                       \
        int                   offset = snprintf(out.data(), out.size(), "in %s %s:%d: ", func, file, line); \
        snprintf(out.data() + offset, out.size() - offset, msg, __VA_ARGS__);                               \
        BatmanInfer::throw_error(BatmanInfer::BIStatus(BatmanInfer::BIErrorCode::RUNTIME_ERROR,             \
                                 std::string(out.data())));                                                 \
    } while (false)

/** Print the given message then throw an std::runtime_error. Accepts a message format
 *  and a variable list of arguments matching the format description.
 *
 * @param[in] msg Error description message format.
 * @param[in] ... List of arguments matching the format description.
 */
#define BI_COMPUTE_ERROR_VAR(msg, ...) BI_COMPUTE_THROW_ERROR_VAR(__func__, __FILE__, __LINE__, msg, __VA_ARGS__)

/** Print the given message then throw an std::runtime_error.
 *
 * @param[in] msg Message to display.
 */
#define BI_COMPUTE_ERROR(msg) BI_COMPUTE_THROW_ERROR(__func__, __FILE__, __LINE__, msg)

/** Print the given message then throw an std::runtime_error. Accepts a message format
 *  and a variable list of arguments matching the format description.
 *
 * @param[in] func Function in which the error occurred.
 * @param[in] file File in which the error occurred.
 * @param[in] line Line in which the error occurred.
 * @param[in] msg  Error description message format.
 * @param[in] ...  List of arguments matching the format description.
 */
#define BI_COMPUTE_ERROR_LOC_VAR(func, file, line, msg, ...) \
    BI_COMPUTE_THROW_ERROR_VAR(func, file, line, msg, __VA_ARGS__) // NOLINT

/** Print the given message then throw an std::runtime_error.
 *
 * @param[in] func Function in which the error occurred.
 * @param[in] file File in which the error occurred.
 * @param[in] line Line in which the error occurred.
 * @param[in] msg  Message to display.
 */
#define BI_COMPUTE_ERROR_LOC(func, file, line, msg) BI_COMPUTE_THROW_ERROR(func, file, line, msg) // NOLINT

/** If the condition is true, the given message is printed and program exits
 *
 * @param[in] cond Condition to evaluate.
 * @param[in] msg  Message to display.
 */
#define BI_COMPUTE_EXIT_ON_MSG(cond, msg)  \
    do                                     \
    {                                      \
        if (cond)                          \
        {                                  \
            BI_COMPUTE_ERROR(msg);        \
        }                                  \
    } while (false)

/** If the condition is true, the given message is printed and program exits. Accepts a message format
*  and a variable list of arguments matching the format description.
*
* @param[in] cond Condition to evaluate.
* @param[in] msg  Error description message format.
* @param[in] ...  List of arguments matching the format description.
*/
#define BI_COMPUTE_EXIT_ON_MSG_VAR(cond, msg, ...)   \
    do                                               \
    {                                                \
        if (cond)                                    \
        {                                            \
            BI_COMPUTE_ERROR_VAR(msg, __VA_ARGS__);  \
        }                                            \
    } while (false)

#ifdef BI_COMPUTE_ASSERTS_ENABLED
/** Checks if a status value is valid if not throws an exception with the error
 *
 * @param[in] status Status value to check.
 */
#define BI_COMPUTE_ERROR_THROW_ON(status) status.throw_if_error()

/** If the condition is true, the given message is printed and an exception is thrown
 *
 * @param[in] cond Condition to evaluate.
 * @param[in] msg  Message to display.
 */
#define BI_COMPUTE_ERROR_ON_MSG(cond, msg) BI_COMPUTE_EXIT_ON_MSG(cond, msg)

/** If the condition is true, the given message is printed and an exception is thrown. Accepts a message format
 *  and a variable list of arguments matching the format description.
 *
 * @param[in] cond Condition to evaluate.
 * @param[in] msg  Error description message format.
 * @param[in] ...  List of arguments matching the format description.
 */
#define BI_COMPUTE_ERROR_ON_MSG_VAR(cond, msg, ...) BI_COMPUTE_EXIT_ON_MSG_VAR(cond, msg, __VA_ARGS__)

/** If the condition is true, the given message is printed and an exception is thrown.
 *
 * @param[in] cond Condition to evaluate.
 * @param[in] func Function in which the error occurred.
 * @param[in] file File in which the error occurred.
 * @param[in] line Line in which the error occurred.
 * @param[in] ...  Message to print if cond is false.
 */
#define BI_COMPUTE_ERROR_ON_LOC_MSG(cond, func, file, line, ...)      \
    do                                                                \
    {                                                                 \
        if (cond)                                                     \
        {                                                             \
            BI_COMPUTE_ERROR_LOC_VAR(func, file, line, __VA_ARGS__);  \
        }                                                             \
    } while (false)

/** If the condition is true, the given message is printed and an exception is thrown, otherwise value is returned
 *
 * @param[in] cond Condition to evaluate.
 * @param[in] val  Value to be returned.
 * @param[in] msg  Message to print if cond is false.
 */
#define BI_COMPUTE_CONST_ON_ERROR(cond, val, msg) (cond) ? throw std::logic_error(msg) : val;
#else /* BI_COMPUTE_ASSERTS_ENABLED */
#define BI_COMPUTE_ERROR_THROW_ON(status)
#define BI_COMPUTE_ERROR_ON_MSG(cond, msg)
#define BI_COMPUTE_ERROR_ON_MSG_VAR(cond, msg, ...)
#define BI_COMPUTE_ERROR_ON_LOC_MSG(cond, func, file, line, ...)
#define BI_COMPUTE_CONST_ON_ERROR(cond, val, msg) val
#endif /* BI_COMPUTE_ASSERTS_ENABLED */

/** If the condition is true then an error message is printed and an exception thrown
 *
 * @param[in] cond Condition to evaluate.
 */
#define BI_COMPUTE_ERROR_ON(cond) BI_COMPUTE_ERROR_ON_MSG(cond, #cond)

/** If the condition is true then an error message is printed and an exception thrown
 *
 * @param[in] cond Condition to evaluate.
 * @param[in] func Function in which the error occurred.
 * @param[in] file File in which the error occurred.
 * @param[in] line Line in which the error occurred.
 */
#define BI_COMPUTE_ERROR_ON_LOC(cond, func, file, line) \
    BI_COMPUTE_ERROR_ON_LOC_MSG(cond, func, file, line, "%s", #cond)

#ifndef BI_COMPUTE_EXCEPTIONS_DISABLED
#define BI_COMPUTE_THROW(ex) throw(ex)
#else /* BI_COMPUTE_EXCEPTIONS_DISABLED */
#define BI_COMPUTE_THROW(ex) (ex), std::abort()
#endif /* BI_COMPUTE_EXCEPTIONS_DISABLED */

#endif //BATMANINFER_BI_CORE_ERROR_H
