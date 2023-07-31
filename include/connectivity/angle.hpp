#ifndef _ANGLE_HPP_

#define _ANGLE_HPP_

#include "connectivityElement.hpp"

namespace connectivity
{
    class Angle;
}   // namespace connectivity

/**
 * @class Angle
 *
 * @brief Represents an angle between three atoms.
 *
 */
class connectivity::Angle : public connectivity::ConnectivityElement
{
  public:
    using connectivity::ConnectivityElement::ConnectivityElement;
};

#endif   // _ANGLE_HPP_