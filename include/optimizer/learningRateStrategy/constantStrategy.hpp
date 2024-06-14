#ifndef _CONSTANT_LEARNING_RATE_STRATEGY_HPP_

#define _CONSTANT_LEARNING_RATE_STRATEGY_HPP_

#include "learningRateStrategy.hpp"

namespace optimization
{
    /**
     * @class ConstantLRStrategy
     *
     * @brief Constant Learning Rate Strategy
     *
     */
    class ConstantLRStrategy : public LearningRateStrategy
    {
       public:
        explicit ConstantLRStrategy(const double initialLearningRate);

        ConstantLRStrategy()           = default;
        ~ConstantLRStrategy() override = default;

        double updateLearningRate() override;
    };

}   // namespace optimization

#endif   // _CONSTANT_LEARNING_RATE_STRATEGY_HPP_