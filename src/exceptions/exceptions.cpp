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
 * @brief Construct a new Custom Exception:: Custom Exception object
 *
 * @param message
 */
CustomException::CustomException(const std::string_view message)
    : _message(message)
{
}

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
const char *InputFileException::what() const noexcept
{
    colorfulOutput(Color::FG_RED, "InputFileError");
    return _message.c_str();
}

/**
 * @brief Construct a new Custom Exception:: Custom Exception object
 *
 * @return const char*
 */
const char *RstFileException::what() const noexcept
{
    colorfulOutput(Color::FG_RED, "RstFileError");
    return _message.c_str();
}

/**
 * @brief Construct a new Custom Exception:: Custom Exception object
 *
 * @return const char*
 */
const char *UserInputException::what() const noexcept
{
    colorfulOutput(Color::FG_RED, "UserInputError");
    return _message.c_str();
}

/**
 * @brief Construct a new Custom Exception:: Custom Exception object
 *
 * @return const char*
 */
const char *MolDescriptorException::what() const noexcept
{
    colorfulOutput(Color::FG_RED, "MolDescriptorError");
    return _message.c_str();
}

/**
 * @brief Construct a new Custom Exception:: Custom Exception object
 *
 * @return const char*
 */
const char *UserInputExceptionWarning::what() const noexcept
{
    colorfulOutput(Color::FG_ORANGE, "UserInputWarning");
    return _message.c_str();
}

/**
 * @brief Construct a new Custom Exception:: Custom Exception object
 *
 * @return const char*
 */
const char *GuffDatException::what() const noexcept
{
    colorfulOutput(Color::FG_RED, "GuffDatError");
    return _message.c_str();
}

/**
 * @brief Construct a new Custom Exception:: Custom Exception object
 *
 * @return const char*
 */
const char *TopologyException::what() const noexcept
{
    colorfulOutput(Color::FG_RED, "TopologyError");
    return _message.c_str();
}

/**
 * @brief Construct a new Custom Exception:: Custom Exception object
 *
 * @return const char*
 */
const char *ParameterFileException::what() const noexcept
{
    colorfulOutput(Color::FG_RED, "ParameterFileError");
    return _message.c_str();
}

/**
 * @brief Construct a new Custom Exception:: Custom Exception object
 *
 * @return const char*
 */
const char *ManostatException::what() const noexcept
{
    colorfulOutput(Color::FG_RED, "ManostatError");
    return _message.c_str();
}

/**
 * @brief Construct a new Custom Exception:: Custom Exception object
 *
 * @return const char*
 */
const char *IntraNonBondedException::what() const noexcept
{
    colorfulOutput(Color::FG_RED, "IntraNonBondedError");
    return _message.c_str();
}

/**
 * @brief Construct a new Custom Exception:: Custom Exception object
 *
 * @return const char*
 */
const char *ShakeException::what() const noexcept
{
    colorfulOutput(Color::FG_RED, "ShakeError");
    return _message.c_str();
}

/**
 * @brief Construct a new Custom Exception:: Custom Exception object
 *
 * @return const char*
 */
const char *CellListException::what() const noexcept
{
    colorfulOutput(Color::FG_RED, "CellListError");
    return _message.c_str();
}

/**
 * @brief Construct a new Custom Exception:: Custom Exception object
 *
 * @param message
 */
const char *RingPolymerRestartFileException::what() const noexcept
{
    colorfulOutput(Color::FG_RED, "RingPolymerRestartFileError");
    return _message.c_str();
}

/**
 * @brief Construct a new Custom Exception:: Custom Exception object
 *
 * @param message
 */
const char *QMRunnerException::what() const noexcept
{
    colorfulOutput(Color::FG_RED, "QMRunnerError");
    return _message.c_str();
}

/**
 * @brief Construct a new Custom Exception:: Custom Exception object
 *
 * @param message
 */
const char *MPIException::what() const noexcept
{
    colorfulOutput(Color::FG_RED, "MPIError");
    return _message.c_str();
}

/**
 * @brief Construct a new Custom Exception:: Custom Exception object
 *
 * @param message
 */
const char *QMRunTimeExceeded::what() const noexcept
{
    colorfulOutput(Color::FG_RED, "QMRunTimeExceeded");
    return _message.c_str();
}

/**
 * @brief Construct a new Custom Exception:: Custom Exception object
 *
 * @param message
 */
const char *MShakeFileException::what() const noexcept
{
    colorfulOutput(Color::FG_RED, "MShakeError");
    return _message.c_str();
}

/**
 * @brief Construct a new Custom Exception:: Custom Exception object
 *
 * @param message
 */
const char *MShakeException::what() const noexcept
{
    colorfulOutput(Color::FG_RED, "MShakeError");
    return _message.c_str();
}

/**
 * @brief Construct a new Custom Exception:: Custom Exception object
 *
 * @param message
 */
const char *LinearAlgebraException::what() const noexcept
{
    colorfulOutput(Color::FG_RED, "LinearAlgebraError");
    return _message.c_str();
}

/**
 * @brief Construct a new Custom Exception:: Custom Exception object
 *
 * @param message
 */
const char *OptException::what() const noexcept
{
    colorfulOutput(Color::FG_RED, "OptimizationError");
    return _message.c_str();
}

/**
 * @brief Construct a new Custom Exception:: Custom Exception object
 *
 * @param message
 */
const char *OptWarning::what() const noexcept
{
    colorfulOutput(Color::FG_ORANGE, "OptimizationWarning");
    return _message.c_str();
}

/**
 * @brief Construct a new Custom Exception:: Custom Exception object
 *
 * @param message
 */
const char *CompileTimeException::what() const noexcept
{
    colorfulOutput(Color::FG_RED, "CompileTimeError");
    return _message.c_str();
}

/**
 * @brief Construct a new Custom Exception:: Custom Exception object
 *
 * @param message
 */
const char *DeviceException::what() const noexcept
{
    colorfulOutput(Color::FG_RED, "DeviceError");
    return _message.c_str();
}

/**
 * @brief Construct a new Custom Exception:: Custom Exception object
 *
 * @param message
 */
const char *NotImplementedException::what() const noexcept
{
    colorfulOutput(Color::FG_RED, "NotImplementedError");
    return _message.c_str();
}