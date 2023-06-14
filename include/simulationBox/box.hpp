#ifndef _BOX_H_

#define _BOX_H_

#include "vector3d.hpp"

/**
 * @class Box
 *
 * @brief
 *
 *  This class stores all the information about the box.
 *
 * @details
 *
 *  This class is used to store the box dimensions and angles.
 *
 */
class Box
{
private:
    Vec3D _boxDimensions = {0.0, 0.0, 0.0};
    Vec3D _boxAngles = {90.0, 90.0, 90.0};

    double _totalMass = 0.0;
    double _totalCharge = 0.0;
    double _density = 0.0;
    double _volume = 0.0;

    bool _boxSizeHasChanged = false;

public:
    void setBoxDimensions(const Vec3D &);
    Vec3D calculateBoxDimensionsFromDensity() const;

    void setBoxAngles(const Vec3D &);

    double calculateVolume() const;

    double calculateDistance(const Vec3D &, const Vec3D &, Vec3D &);
    double calculateDistanceSquared(const Vec3D &, const Vec3D &, Vec3D &);

    void applyPBC(Vec3D &) const;

    void scaleBox(const Vec3D &);

    [[nodiscard]] double getMinimalBoxDimension() const { return minimum(_boxDimensions); };

    // standard getter and setters
    [[nodiscard]] Vec3D getBoxDimensions() const { return _boxDimensions; };
    [[nodiscard]] Vec3D getBoxAngles() const { return _boxAngles; };

    void setTotalMass(const double totalMass) { _totalMass = totalMass; };
    [[nodiscard]] double getTotalMass() const { return _totalMass; };

    void setTotalCharge(const double totalCharge) { _totalCharge = totalCharge; };
    [[nodiscard]] double getTotalCharge() const { return _totalCharge; };

    void setDensity(const double density);
    [[nodiscard]] double getDensity() const { return _density; };

    void setVolume(const double volume) { _volume = volume; };
    [[nodiscard]] double getVolume() const { return _volume; };

    void setBoxSizeHasChanged(const bool boxSizeHasChanged) { _boxSizeHasChanged = boxSizeHasChanged; };
    [[nodiscard]] bool getBoxSizeHasChanged() const { return _boxSizeHasChanged; };
};

#endif