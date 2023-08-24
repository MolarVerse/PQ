#ifndef _ANGLE_HPP_

#define _ANGLE_HPP_

#include "connectivityElement.hpp"

namespace connectivity
{
    /**
     * @class Angle
     *
     * @brief Represents an angle between three atoms.
     *
     */
    class Angle : public ConnectivityElement
    {
      public:
        using ConnectivityElement::ConnectivityElement;
    };

}   // namespace connectivity

#endif   // _ANGLE_HPP_