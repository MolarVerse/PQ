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

#ifndef _EXCEPTIONS_HPP_

#define _EXCEPTIONS_HPP_

#include "baseException.hpp"
#include "color.hpp"

namespace exc
{
    using InputFileException =
        BaseException<Color::FG_RED, ExceptionType::InputFileError>;

    using RstFileException =
        BaseException<Color::FG_RED, ExceptionType::RstFileError>;

    using UserInputException =
        BaseException<Color::FG_RED, ExceptionType::UserInputError>;

    using MolDescriptorException =
        BaseException<Color::FG_RED, ExceptionType::MoldescriptorError>;

    using UserInputExceptionWarning =
        BaseException<Color::FG_ORANGE, ExceptionType::UserInputWarning>;

    using GuffDatException =
        BaseException<Color::FG_RED, ExceptionType::GuffDatError>;

    using TopologyException =
        BaseException<Color::FG_RED, ExceptionType::TopologyError>;

    using ParameterFileException =
        BaseException<Color::FG_RED, ExceptionType::ParameterFileError>;

    using ManostatException =
        BaseException<Color::FG_RED, ExceptionType::ManostatError>;

    using IntraNonBondedException =
        BaseException<Color::FG_RED, ExceptionType::IntraNonBondedError>;

    using ShakeException =
        BaseException<Color::FG_RED, ExceptionType::ShakeError>;

    using CellListException =
        BaseException<Color::FG_RED, ExceptionType::CellListError>;

    using RingPolymerRestartFileException = BaseException<
        Color::FG_RED,
        ExceptionType::RingPolymerRestartFileError>;

    using QMRunnerException =
        BaseException<Color::FG_RED, ExceptionType::QmRunnerError>;

    using MPIException = BaseException<Color::FG_RED, ExceptionType::MpiError>;

    using QMRunTimeExceeded =
        BaseException<Color::FG_RED, ExceptionType::QmRuntimeExceeded>;

    using MShakeFileException =
        BaseException<Color::FG_RED, ExceptionType::MShakeFileError>;

    using MShakeException =
        BaseException<Color::FG_RED, ExceptionType::MShakeError>;

    using LinearAlgebraException =
        BaseException<Color::FG_RED, ExceptionType::LinearAlgebraError>;

    using OptException =
        BaseException<Color::FG_RED, ExceptionType::OptimizationError>;

    using OptWarning =
        BaseException<Color::FG_ORANGE, ExceptionType::OptimizationWarning>;

    using CompileTimeException =
        BaseException<Color::FG_RED, ExceptionType::CompileTimeError>;

    using HybridConfiguratorException =
        BaseException<Color::FG_RED, ExceptionType::HybridConfiguratorError>;

    using HybridMDEngineException =
        BaseException<Color::FG_RED, ExceptionType::HybridMDEngineError>;

    using TimerException =
        BaseException<Color::FG_RED, ExceptionType::TimerError>;

}   // namespace exc

#endif   // _EXCEPTIONS_HPP_
