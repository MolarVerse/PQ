/*****************************************************************************
<GPL_HEADER>

    PQ
    Copyright (C) 2023-now  Jakob Gamper

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

<GPL_HEADER>
******************************************************************************/

#include "exceptions.hpp"

#include <iostream>

using namespace customException;

/**
 * @brief Prints the exception type in color.
 *
 * @param color
 * @param exception
 */
void CustomException::colorfulOutput(
    const Color::Code      color,
    const std::string_view exception
) const
{
    const Color::Modifier modifier(color);
    const Color::Modifier def(Color::FG_DEFAULT);

    std::cout << modifier << exception << def << '\n' << std::flush;
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

/**
 * @brief Construct a new Custom Exception:: Custom Exception object
 *
 * @param message
 */
const char *RingPolymerRestartFileException::what() const throw()
{
    colorfulOutput(Color::FG_RED, "RingPolymerRestartFileError");
    return _message.c_str();
}

/**
 * @brief Construct a new Custom Exception:: Custom Exception object
 *
 * @param message
 */
const char *QMRunnerException::what() const throw()
{
    colorfulOutput(Color::FG_RED, "QMRunnerError");
    return _message.c_str();
}

/**
 * @brief Construct a new Custom Exception:: Custom Exception object
 *
 * @param message
 */
const char *MPIException::what() const throw()
{
    colorfulOutput(Color::FG_RED, "MPIError");
    return _message.c_str();
}

/**
 * @brief Construct a new Custom Exception:: Custom Exception object
 *
 * @param message
 */
const char *QMRunTimeExceeded::what() const throw()
{
    colorfulOutput(Color::FG_RED, "QMRunTimeExceeded");
    return _message.c_str();
}

/**
 * @brief Construct a new Custom Exception:: Custom Exception object
 *
 * @param message
 */
const char *MShakeFileException::what() const throw()
{
    colorfulOutput(Color::FG_RED, "MShakeError");
    return _message.c_str();
}

/**
 * @brief Construct a new Custom Exception:: Custom Exception object
 *
 * @param message
 */
const char *MShakeException::what() const throw()
{
    colorfulOutput(Color::FG_RED, "MShakeError");
    return _message.c_str();
}

/**
 * @brief Construct a new Custom Exception:: Custom Exception object
 *
 * @param message
 */
const char *LinearAlgebraException::what() const throw()
{
    colorfulOutput(Color::FG_RED, "LinearAlgebraError");
    return _message.c_str();
}