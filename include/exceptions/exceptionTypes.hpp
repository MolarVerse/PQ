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

#ifndef _EXCEPTION_TYPES_HPP_
#define _EXCEPTION_TYPES_HPP_

#include <cstdint>
#include <mstd/enum.hpp>

#define EXCEPTION_TYPES(X)         \
    X(Undefined)                   \
    X(InputFileError)              \
    X(RstFileError)                \
    X(UserInputError)              \
    X(MoldescriptorError)          \
    X(UserInputWarning)            \
    X(GuffDatError)                \
    X(TopologyError)               \
    X(ParameterFileError)          \
    X(ManostatError)               \
    X(IntraNonBondedError)         \
    X(ShakeError)                  \
    X(CellListError)               \
    X(RingPolymerRestartFileError) \
    X(QmRunnerError)               \
    X(MpiError)                    \
    X(QmRuntimeExceeded)           \
    X(MShakeFileError)             \
    X(MShakeError)                 \
    X(LinearAlgebraError)          \
    X(OptimizationError)           \
    X(OptimizationWarning)         \
    X(CompileTimeError)            \
    X(HybridConfiguratorError)     \
    X(HybridMDEngineError)         \
    X(TimerError)

MSTD_ENUM(ExceptionType, std::uint8_t, EXCEPTION_TYPES)

#endif   // _EXCEPTION_TYPES_HPP_
