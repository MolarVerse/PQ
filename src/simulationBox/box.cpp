#include <vector>
#include <string>
#include <stdexcept>
#include <iostream>

#include "box.hpp"

using namespace std;

Box::Box() : _boxDimensions({0.0, 0.0, 0.0}), _boxAngles({0.0, 0.0, 0.0}) {}
Box::~Box() {}

vector<double> Box::getBoxDimensions()
{
    return _boxDimensions;
}

/**
 * @brief Set the Box Dimensions in Box object
 * 
 * @param boxDimensions 
 * 
 * @throw range_error if any of the dimensions is negative
 */
void Box::setBoxDimensions(const vector<double> &boxDimensions)
{
    for (auto &dimension : boxDimensions)
        if (dimension < 0.0)
            throw range_error("Box dimensions must be positive - dimension = " + to_string(dimension));

    _boxDimensions = boxDimensions;
}

vector<double> Box::getBoxAngles()
{
    return _boxAngles;
}

/**
 * @brief Set the Box Angles in Box object
 * 
 * @param boxAngles 
 * 
 * @throw range_error if any of the angles is negative or greater than 90°
 */
void Box::setBoxAngles(const vector<double> &boxAngles)
{
    for (auto &angle : boxAngles)
        if (angle < 0.0 || angle > 90.0)
            throw range_error("Box angles must be positive and smaller than 90° - angle = " + to_string(angle));

    _boxAngles = boxAngles;
}