#include "outputData.hpp"
#include "mathUtilities.hpp"

void OutputData::setMomentumVector(const std::vector<double> &momentumVector)
{
    _momentumVector = momentumVector;
    _momentum = MathUtilities::norm(_momentumVector);
    _averageMomentum += _momentum;
}