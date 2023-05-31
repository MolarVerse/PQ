#include "mathUtilities.hpp"

#include <algorithm>
#include <vector>
#include <cmath>

using namespace std;

/**
 * @brief calculates the norm of a vector
 *
 * @param vector
 * @return double
 */
double MathUtilities::norm(const std::vector<double> &vector)
{
    double norm = 0.0;
    for_each(vector.begin(), vector.end(), [&norm](const double value)
             { norm += value * value; });
    return sqrt(norm);
}