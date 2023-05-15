#ifndef _OUTPUT_DATA_H_

#define _OUTPUT_DATA_H_

#include <vector>

/**
 * @class OutputData
 *
 * @brief OutputData is a class for output data storage
 *
 */
class OutputData
{
private:
    std::vector<double> _momentumVector = {0.0, 0.0, 0.0};
    double _momentum = 0.0;
    double _averageMomentum = 0.0;

public:
    void setMomentumVector(const std::vector<double> &momentumVector);

    double getMomentum() const { return _momentum; }
    double getAverageMomentum() const { return _averageMomentum; }
};

#endif // _OUTPUT_DATA_H_
