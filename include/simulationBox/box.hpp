#ifndef _BOX_H_

#define _BOX_H_

#include <vector>

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
    std::vector<double> _boxDimensions = {0.0, 0.0, 0.0};
    std::vector<double> _boxAngles = {90.0, 90.0, 90.0};

    double _totalMass = 0.0;
    double _totalCharge = 0.0;
    double _density = 0.0;
    double _volume = 0.0;

    bool _boxSizeHasChanged = false;

public:
    void setBoxDimensions(const std::vector<double> &);
    std::vector<double> calculateBoxDimensionsFromDensity() const;

    void setBoxAngles(const std::vector<double> &);

    double calculateVolume() const;

    double calculateDistance(const std::vector<double> &, const std::vector<double> &, std::vector<double> &);
    double calculateDistanceSquared(const std::vector<double> &, const std::vector<double> &, std::vector<double> &);

    void applyPBC(std::vector<double> &);

    double getMinimalBoxDimension() const;

    // standard getter and setters
    std::vector<double> getBoxDimensions() const { return _boxDimensions; };
    std::vector<double> getBoxAngles() const { return _boxAngles; };

    void setTotalMass(double totalMass) { _totalMass = totalMass; };
    double getTotalMass() const { return _totalMass; };

    void setTotalCharge(double totalCharge) { _totalCharge = totalCharge; };
    double getTotalCharge() const { return _totalCharge; };

    double getDensity() const { return _density; };
    void setDensity(double density);

    double getVolume() const { return _volume; };
    void setVolume(double volume) { _volume = volume; };

    bool getBoxSizeHasChanged() const { return _boxSizeHasChanged; };
    void setBoxSizeHasChanged(bool boxSizeHasChanged) { _boxSizeHasChanged = boxSizeHasChanged; };
};

#endif