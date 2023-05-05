#include "exceptions.hpp"

/**
 * @brief Prints the exception type in color.
 *
 * @param color
 * @param exception
 */
void CustomException::colorfulOutput(Color::Code color, std::string_view exception) const
{
    Color::Modifier modifier(color);
    Color::Modifier def(Color::FG_DEFAULT);

    std::cout << modifier << exception << def << std::endl;
}

/**
 * @brief Construct a new Custom Exception:: Custom Exception object
 *
 * @return const char*
 */
const char *InputFileException::what() const throw()
{
    colorfulOutput(Color::FG_RED, "InputFileError");
    return _message.c_str();
}

/**
 * @brief Construct a new Custom Exception:: Custom Exception object
 *
 * @return const char*
 */
const char *RstFileException::what() const throw()
{
    colorfulOutput(Color::FG_RED, "RstFileError");
    return _message.c_str();
}

/**
 * @brief Construct a new Custom Exception:: Custom Exception object
 *
 * @return const char*
 */
const char *UserInputException::what() const throw()
{
    colorfulOutput(Color::FG_RED, "UserInputError");
    return _message.c_str();
}