#include "outputData.hpp"
#include "mathUtilities.hpp"

/**
 * @brief sets the momentum vector and calculates the absolute box momentum
 *
 * @param momentumVector
 */
void OutputData::setMomentumVector(const std::vector<double> &momentumVector)
{
    _momentumVector = momentumVector;
    _momentum = MathUtilities::norm(_momentumVector);
    _averageMomentum += _momentum;
}