#include "exceptions.hpp"

#include <iostream>

using namespace std;
using namespace customException;

/**
 * @brief Prints the exception type in color.
 *
 * @param color
 * @param exception
 */
void CustomException::colorfulOutput(const Color::Code color, const string_view exception) const
{
    const Color::Modifier modifier(color);
    const Color::Modifier def(Color::FG_DEFAULT);

    cout << modifier << exception << def << '\n' << std::flush;
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

/**
 * @brief Construct a new Custom Exception:: Custom Exception object
 *
 * @return const char*
 */
const char *MolDescriptorException::what() const throw()
{
    colorfulOutput(Color::FG_RED, "MolDescriptorError");
    return _message.c_str();
}

/**
 * @brief Construct a new Custom Exception:: Custom Exception object
 *
 * @return const char*
 */
const char *UserInputExceptionWarning::what() const throw()
{
    colorfulOutput(Color::FG_ORANGE, "UserInputWarning");
    return _message.c_str();
}

/**
 * @brief Construct a new Custom Exception:: Custom Exception object
 *
 * @return const char*
 */
const char *GuffDatException::what() const throw()
{
    colorfulOutput(Color::FG_RED, "GuffDatError");
    return _message.c_str();
}

/**
 * @brief Construct a new Custom Exception:: Custom Exception object
 *
 * @return const char*
 */
const char *TopologyException::what() const throw()
{
    colorfulOutput(Color::FG_RED, "TopologyError");
    return _message.c_str();
}

/**
 * @brief Construct a new Custom Exception:: Custom Exception object
 *
 * @return const char*
 */
const char *ParameterFileException::what() const throw()
{
    colorfulOutput(Color::FG_RED, "ParameterFileError");
    return _message.c_str();
}

/**
 * @brief Construct a new Custom Exception:: Custom Exception object
 *
 * @return const char*
 */
const char *ManostatException::what() const throw()
{
    colorfulOutput(Color::FG_RED, "ManostatError");
    return _message.c_str();
}

/**
 * @brief Construct a new Custom Exception:: Custom Exception object
 *
 * @return const char*
 */
const char *IntraNonBondedException::what() const throw()
{
    colorfulOutput(Color::FG_RED, "IntraNonBondedError");
    return _message.c_str();
}

/**
 * @brief Construct a new Custom Exception:: Custom Exception object
 *
 * @return const char*
 */
const char *ShakeException::what() const throw()
{
    colorfulOutput(Color::FG_RED, "ShakeError");
    return _message.c_str();
}

/**
 * @brief Construct a new Custom Exception:: Custom Exception object
 *
 * @return const char*
 */
const char *CellListException::what() const throw()
{
    colorfulOutput(Color::FG_RED, "CellListError");
    return _message.c_str();
}