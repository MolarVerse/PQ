#ifndef _OPTIMIZER_TPP_

#define _OPTIMIZER_TPP_

#include "optimizer.hpp"

namespace optimization
{
    /**
     * @brief make shared_ptr for learning rate strategy
     *
     * @tparam T
     * @param strategy
     */
    template <typename T>
    inline void Optimizer::makeLearningRateStrategy(T strategy)
    {
        _learningRateStrategy = std::make_shared<T>(strategy);
    }
}   // namespace optimization

#endif   // _OPTIMIZER_TPP_