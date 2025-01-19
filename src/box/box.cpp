/*****************************************************************************
<GPL_HEADER>

    PQ
    Copyright (C) 2023-now  Jakob Gamper

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

<GPL_HEADER>
******************************************************************************/

/**
 * @file box.cpp
 * @author Jakob Gamper (97gamjak@gmail.com)
 * @brief This file contains the implementation of the box class. The box class
 * is the base class for all box types. It contains the basic methods which are
 * needed for all box types.
 *
 * @date 2024-12-09
 *
 * @see box.hpp
 */

#include "box.hpp"

#include <memory>

#include "settings.hpp"

#ifdef __PQ_GPU__
    #include "device.hpp"
#endif

using namespace linearAlgebra;
using namespace simulationBox;
using namespace settings;
using namespace device;

/******************************************************
 *                                                    *
 * virtual methods that are overriden in triclinicBox *
 *                                                    *
 ******************************************************/

/**
 * @brief get the box angles
 *
 * @return Vec3D
 */
Vec3D Box::getBoxAngles() const { return Vec3D(90.0); }

/**
 * @brief get the box matrix
 *
 * @return StaticMatrix3x3<double>
 */
StaticMatrix3x3<double> Box::getBoxMatrix() const
{
    return diagonalMatrix(_boxDimensions);
}

/**
 * @brief transform a vector into orthogonal space
 *
 * @param position
 * @return Vec3D
 */
Vec3D Box::toOrthoSpace(const Vec3D &position) const { return position; }

/**
 * @brief transform a tensor into orthogonal space
 *
 * @param position
 * @return tensor3D
 */
tensor3D Box::toOrthoSpace(const tensor3D &position) const { return position; }

/**
 * @brief transform a vector into simulation space
 *
 * @param position
 * @return Vec3D
 */
Vec3D Box::toSimSpace(const Vec3D &position) const { return position; }

/**
 * @brief transform a tensor into simulation space
 *
 * @param position
 * @return tensor3D
 */
tensor3D Box::toSimSpace(const tensor3D &position) const { return position; }

/**
 * @brief set the box dimensions
 *
 * @param boxDimensions
 */
void Box::setBoxDimensions(const Vec3D &boxDimensions)
{
    _boxDimensions = boxDimensions;
}

/***************************
 *                         *
 * standard getter methods *
 *                         *
 ***************************/

/**
 * @brief get if the box size has changed
 *
 * @return true
 * @return false
 */
bool Box::getBoxSizeHasChanged() const { return _boxSizeHasChanged; }

/**
 * @brief get the volume of the box
 *
 * @return double
 */
double Box::getVolume() const { return _volume; }

/**
 * @brief get the minimal box dimension
 *
 * @return double
 */
double Box::getMinimalBoxDimension() const { return minimum(_boxDimensions); }

/**
 * @brief get the box dimensions
 *
 * @return Vec3D
 */
linearAlgebra::Vec3D Box::getBoxDimensions() const { return _boxDimensions; }

/**
 * @brief get the box params
 *
 * @details The box params are the parameters of the box which are needed for
 * the minimum image convention. The box params are stored in a vector. If a
 * device is used the box params are stored on the device as a pointer. The
 * method returns the pointer to the box params either on the device or on the
 * host.
 *
 * @return Real*
 *
 */
Real *Box::getBoxParamsPtr()
{
#ifdef __PQ_GPU__
    if (Settings::useDevice())
        return _boxParamsDevice;
    else
#endif
        return _boxParams.data();
}

#ifdef __PQ_GPU__
/**
 * @brief copy the box params to the device
 *
 * @param device
 * @return Real*
 */
void Box::copyBoxParamsTo()
{
    _device->deviceMemcpyToAsync(_boxParamsDevice, _boxParams);
    _device->checkErrors("Box copy box params to device");
}

/**
 * @brief copy the box params from the device
 *
 * @param device
 * @return Real*
 */
void Box::copyBoxParamsFrom()
{
    _device->deviceMemcpyFromAsync(_boxParams, _boxParamsDevice);
    _device->checkErrors("Box copy box params from device");
}

/**
 * @brief initialize the device memory
 *
 * @param device
 */
void Box::initDeviceMemory(Device &device)
{
    _device = std::make_shared<Device>(device);

    device.deviceMalloc(&_boxParamsDevice, 18);
    device.checkErrors("Box device memory allocation");
}

#endif

/********************
 * standard setters *
 ********************/

/**
 * @brief set the volume of the box
 *
 * @param volume
 */
void Box::setVolume(const double volume) { _volume = volume; }

/**
 * @brief set if the box size has changed
 *
 * @param boxSizeHasChanged
 */
void Box::setBoxSizeHasChanged(const bool boxSizeHasChanged)
{
    _boxSizeHasChanged = boxSizeHasChanged;
}