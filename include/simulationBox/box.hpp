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
    std::vector<double> _boxDimensions;
    std::vector<double> _boxAngles;
public:
    Box();
    ~Box();

    std::vector<double> getBoxDimensions();
    void setBoxDimensions(const std::vector<double> &);

    std::vector<double> getBoxAngles();
    void setBoxAngles(const std::vector<double> &);
};

#endif