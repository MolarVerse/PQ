#ifndef _OPTIMIZER_ENUM_HPP_
#define _OPTIMIZER_ENUM_HPP_

#include <cstdint>
#include <mstd/enum.hpp>

/**
 * @brief enum OptimizerType
 *
 */
enum class OptimizerType : std::uint8_t;

#define OPTIMIZER_TYPE_LIST(X) \
    X(NONE)                    \
    X(STEEPEST_DESCENT)        \
    X(ADAM)

MSTD_ENUM(OptimizerType, std::uint8_t, OPTIMIZER_TYPE_LIST)

#undef OPTIMIZER_TYPE_LIST

/**
 * @brief enum LearningRate
 *
 */
enum class LearningRate : std::uint8_t;

#define LEARNING_RATE_LIST(X) \
    X(NONE)                   \
    X(CONSTANT)               \
    X(CONSTANT_DECAY)         \
    X(EXPONENTIAL_DECAY)      \
    X(LINESEARCH_WOLFE)

MSTD_ENUM(LearningRate, std::uint8_t, LEARNING_RATE_LIST)

#undef LEARNING_RATE_LIST

#endif   // _OPTIMIZER_ENUM_HPP_
