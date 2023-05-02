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
    std::vector<double> _boxAngles = {0.0, 0.0, 0.0};

public:
    Box() = default;
    ~Box() = default;

    std::vector<double> getBoxDimensions() const;
    void setBoxDimensions(const std::vector<double> &);

    std::vector<double> getBoxAngles() const;
    void setBoxAngles(const std::vector<double> &);
};

#endif