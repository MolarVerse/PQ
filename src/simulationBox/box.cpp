#include <vector>
#include <string>
#include <stdexcept>
#include <iostream>
#include <cmath>

#include "box.hpp"
#include "exceptions.hpp"
#include "constants.hpp"

using namespace std;

/**
 * @brief Set the Box Dimensions in Box object
 *
 * @param boxDimensions
 *
 * @throw RstFileException if any of the dimensions is negative
 */
void Box::setBoxDimensions(const vector<double> &boxDimensions)
{
    for (auto &dimension : boxDimensions)
        if (dimension < 0.0)
            throw RstFileException("Box dimensions must be positive - dimension = " + to_string(dimension));

    _boxDimensions = boxDimensions;
}

/**
 * @brief Set the Box Angles in Box object
 *
 * @param boxAngles
 *
 * @throw RstFileException if any of the angles is negative or greater than 90°
 */
void Box::setBoxAngles(const vector<double> &boxAngles)
{
    for (auto &angle : boxAngles)
        if (angle < 0.0 || angle > 90.0)
            throw RstFileException("Box angles must be positive and smaller than 90° - angle = " + to_string(angle));

    _boxAngles = boxAngles;
}

/**
 * @brief Set the Density in Box object
 *
 * @param density
 *
 * @throw InputFileException if density is negative
 */
void Box::setDensity(double density)
{
    if (density < 0.0)
        throw InputFileException("Density must be positive - density = " + to_string(density));

    _density = density;
}

/**
 * @brief Calculate the volume of the box
 *
 * @details
 *  The volume is calculated using the formula:
 *  V = a * b * c * sqrt(1 - cos(alpha)^2 - cos(beta)^2 - cos(gamma)^2 + 2 * cos(alpha) * cos(beta) * cos(gamma))
 *  where a, b, c are the box dimensions and alpha, beta, gamma are the box angles.
 *
 *  The volume is stored in the _volume attribute and returned.
 *  The density is also calculated and stored in the _density attribute.
 *
 * @return volume
 */
double Box::calculateVolume() const
{
    double volume = _boxDimensions[0] * _boxDimensions[1] * _boxDimensions[2];

    double cos_alpha = cos(_boxAngles[0] * M_PI / 180.0);
    double cos_beta = cos(_boxAngles[1] * M_PI / 180.0);
    double cos_gamma = cos(_boxAngles[2] * M_PI / 180.0);

    volume *= sqrt(1.0 - cos_alpha * cos_alpha - cos_beta * cos_beta - cos_gamma * cos_gamma + 2.0 * cos_alpha * cos_beta * cos_gamma);

    return volume;
}

/**
 * @brief Calculate the box dimensions from the density
 *
 * @return vector<double>
 */
vector<double> Box::calculateBoxDimensionsFromDensity() const
{
    double volume = _totalMass / (_density * _KG_PER_LITER_CUBIC_TO_AMU_PER_ANGSTROM_CUBIC_);
    double a = cbrt(volume);
    double b = cbrt(volume);
    double c = cbrt(volume);

    vector<double> boxDimensions = {a, b, c};

    return boxDimensions;
}

double Box::calculateDistance(const vector<double> &point1, const vector<double> &point2, vector<double> &dxyz)
{
    dxyz[0] = point1[0] - point2[0];
    dxyz[1] = point1[1] - point2[1];
    dxyz[2] = point1[2] - point2[2];

    applyPBC(dxyz);

    double distance = sqrt(dxyz[0] * dxyz[0] + dxyz[1] * dxyz[1] + dxyz[2] * dxyz[2]);

    return distance;
}

double Box::calculateDistanceSquared(const vector<double> &point1, const vector<double> &point2, vector<double> &dxyz)
{
    dxyz[0] = point1[0] - point2[0];
    dxyz[1] = point1[1] - point2[1];
    dxyz[2] = point1[2] - point2[2];

    applyPBC(dxyz);

    double distanceSquared = dxyz[0] * dxyz[0] + dxyz[1] * dxyz[1] + dxyz[2] * dxyz[2];

    return distanceSquared;
}

void Box::applyPBC(vector<double> &dxyz)
{
    dxyz[0] -= _boxDimensions[0] * round(dxyz[0] / _boxDimensions[0]);
    dxyz[1] -= _boxDimensions[1] * round(dxyz[1] / _boxDimensions[1]);
    dxyz[2] -= _boxDimensions[2] * round(dxyz[2] / _boxDimensions[2]);
}

double Box::getMinimalBoxDimension() const
{
    double minDimension = _boxDimensions[0];
    if (_boxDimensions[1] < minDimension)
        minDimension = _boxDimensions[1];
    if (_boxDimensions[2] < minDimension)
        minDimension = _boxDimensions[2];
    return minDimension;
}