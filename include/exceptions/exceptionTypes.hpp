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
