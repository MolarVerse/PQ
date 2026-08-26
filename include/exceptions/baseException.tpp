#ifndef _BASE_EXCEPTION_TPP_
#define _BASE_EXCEPTION_TPP_

#include <iostream>

#include "baseException.hpp"

namespace exc
{
    /**
     * @brief Construct a new Custom Exception:: Custom Exception object
     *
     * @param message
     */
    template <Color::Code colorCode, ExceptionType exceptionType>
    BaseException<colorCode, exceptionType>::BaseException(
        const std::string_view message,
        std::optional<size_t>  lineNumber
    )
        : PQException(message, lineNumber)
    {
    }

    /**
     * @brief Construct a new Custom Exception:: Custom Exception object
     *
     * @param message
     */
    template <Color::Code colorCode, ExceptionType exceptionType>
    BaseException<colorCode, exceptionType>::BaseException(
        const std::string_view message
    )
        : PQException(message, std::nullopt)
    {
    }

    /**
     * @brief Prints the exceptionMsg type in color.
     *
     * @param color
     * @param exceptionMsg
     */
    template <Color::Code colorCode, ExceptionType exceptionType>
    void BaseException<colorCode, exceptionType>::colorfulOutput(
        const Color::Code      color,
        const std::string_view exceptionMsg
    )
    {
        const Color::Modifier modifier(color);
        const Color::Modifier def(Color::FG_DEFAULT);

        std::cout << modifier << exceptionMsg << def << '\n' << std::flush;
    }

    /**
     * @brief Construct a new Custom Exception:: Custom Exception object
     *
     * @return const char*
     */
    template <Color::Code colorCode, ExceptionType exceptionType>
    const char *BaseException<colorCode, exceptionType>::what() const noexcept
    {
        if (exceptionType != ExceptionType::Undefined)
        {
            colorfulOutput(
                colorCode,
                ExceptionTypeMeta::toString(exceptionType)
            );
        }

        return PQException::getMessage().c_str();
    }
}   // namespace exc

#endif   // _BASE_EXCEPTION_TPP_
