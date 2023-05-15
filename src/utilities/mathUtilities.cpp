#include "mathUtilities.hpp"

#include <vector>
#include <cmath>

using namespace std;

double MathUtilities::norm(const std::vector<double> &vector)
{
    double norm = 0.0;
    for (size_t i = 0; i < vector.size(); i++)
    {
        norm += vector[i] * vector[i];
    }
    return sqrt(norm);
}