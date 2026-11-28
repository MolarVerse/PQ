#include "exceptions.hpp"

namespace exc
{
    /**
     * @brief Constructs an InternalException with a message
     *
     * Appends a report hint to the provided message.
     */
    InternalException::InternalException(std::string_view message)
        : BaseException(std::string(message) + std::string(_reportHint))
    {
    }

    /**
     * @brief Constructs an InternalException with a message and an optional
     * line number
     *
     * Appends a report hint to the provided message.
     */
    InternalException::InternalException(
        std::string_view      message,
        std::optional<size_t> lineNumber
    )
        : BaseException(
              std::string(message) + std::string(_reportHint),
              lineNumber
          )
    {
    }
}   // namespace exc
