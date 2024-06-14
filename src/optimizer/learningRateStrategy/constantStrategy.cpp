#include "constantStrategy.hpp"

using namespace optimization;

/**
 * @brief Construct a new ConstantLRStrategy::ConstantLRStrategy object
 *
 * @param initialLearningRate
 */
ConstantLRStrategy::ConstantLRStrategy(const double initialLearningRate)
    : LearningRateStrategy(initialLearningRate)
{
}

/**
 * @brief Update the learning rate
 *
 * @return double
 */
double ConstantLRStrategy::updateLearningRate() { return _initialLearningRate; }