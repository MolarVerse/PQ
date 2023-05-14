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

public:
    Box() = default;
    ~Box() = default;

    std::vector<double> getBoxDimensions() const { return _boxDimensions; };
    void setBoxDimensions(const std::vector<double> &);
    std::vector<double> calculateBoxDimensionsFromDensity() const;

    std::vector<double> getBoxAngles() const { return _boxAngles; };
    void setBoxAngles(const std::vector<double> &);

    void setTotalMass(double totalMass) { _totalMass = totalMass; };
    double getTotalMass() const { return _totalMass; };

    void setTotalCharge(double totalCharge) { _totalCharge = totalCharge; };
    double getTotalCharge() const { return _totalCharge; };

    double getDensity() const { return _density; };
    void setDensity(double density);

    double getVolume() const { return _volume; };
    void setVolume(double volume) { _volume = volume; };
    double calculateVolume() const;
};

#endif