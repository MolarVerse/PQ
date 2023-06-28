#ifndef _BOX_H_

#define _BOX_H_

#include "vector3d.hpp"

namespace simulationBox
{
    class Box;
}

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
class simulationBox::Box
{
  private:
    vector3d::Vec3D _boxDimensions;
    vector3d::Vec3D _boxAngles = {90.0, 90.0, 90.0};

    double _totalMass;
    double _totalCharge;
    double _density;
    double _volume;

    bool _boxSizeHasChanged = false;

  public:
    double calculateVolume();

    vector3d::Vec3D calculateBoxDimensionsFromDensity() const;

    void applyPBC(vector3d::Vec3D &) const;
    void scaleBox(const vector3d::Vec3D &);

    /***********************************
     * non-standard getter and setters *
     ***********************************/

    void setBoxDimensions(const vector3d::Vec3D &);
    void setBoxAngles(const vector3d::Vec3D &);
    void setDensity(const double density);

    double getMinimalBoxDimension() const { return minimum(_boxDimensions); }

    /*******************************
     * standard getter and setters *
     *******************************/

    vector3d::Vec3D getBoxDimensions() const { return _boxDimensions; }
    vector3d::Vec3D getBoxAngles() const { return _boxAngles; }
    double          getTotalMass() const { return _totalMass; }
    double          getTotalCharge() const { return _totalCharge; }
    double          getDensity() const { return _density; }
    double          getVolume() const { return _volume; }
    bool            getBoxSizeHasChanged() const { return _boxSizeHasChanged; }

    void setTotalMass(const double totalMass) { _totalMass = totalMass; }
    void setTotalCharge(const double totalCharge) { _totalCharge = totalCharge; }
    void setVolume(const double volume) { _volume = volume; }
    void setBoxSizeHasChanged(const bool boxSizeHasChanged) { _boxSizeHasChanged = boxSizeHasChanged; }
};

#endif