#ifndef _ANGLE_TYPE_HPP_

#define _ANGLE_TYPE_HPP_

#include <cstddef>

namespace forceField
{
    /**
     * @class AngleType
     *
     * @brief represents an angle type
     *
     * @details this is a class representing an angle type defined in the parameter file
     *
     */
    class AngleType
    {
      private:
        size_t _id;

        double _equilibriumAngle;
        double _forceConstant;

      public:
        AngleType(size_t id, double equilibriumAngle, double springConstant)
            : _id(id), _equilibriumAngle(equilibriumAngle), _forceConstant(springConstant){};

        [[nodiscard]] bool operator==(const AngleType &other) const;

        [[nodiscard]] size_t getId() const { return _id; }
        [[nodiscard]] double getEquilibriumAngle() const { return _equilibriumAngle; }
        [[nodiscard]] double getForceConstant() const { return _forceConstant; }
    };

}   // namespace forceField

#endif   // _ANGLE_TYPE_HPP_