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

    double _coulombEnergy = 0.0;
    double _averageCoulombEnergy = 0.0;

    double _nonCoulombEnergy = 0.0;
    double _averageNonCoulombEnergy = 0.0;

public:
    void setMomentumVector(const std::vector<double> &momentumVector);

    double getMomentum() const { return _momentum; }
    double getAverageMomentum() const { return _averageMomentum; }

    void setCoulombEnergy(double coulombEnergy) { _coulombEnergy = coulombEnergy; }
    double getCoulombEnergy() const { return _coulombEnergy; }

    void setAverageCoulombEnergy(double averageCoulombEnergy) { _averageCoulombEnergy = averageCoulombEnergy; }
    void addAverageCoulombEnergy(double averageCoulombEnergy) { _averageCoulombEnergy += averageCoulombEnergy; }
    double getAverageCoulombEnergy() const { return _averageCoulombEnergy; }

    void setNonCoulombEnergy(double nonCoulombEnergy) { _nonCoulombEnergy = nonCoulombEnergy; }
    double getNonCoulombEnergy() const { return _nonCoulombEnergy; }

    void setAverageNonCoulombEnergy(double nonCoulombEnergy) { _averageNonCoulombEnergy = nonCoulombEnergy; }
    void addAverageNonCoulombEnergy(double nonCoulombEnergy) { _averageNonCoulombEnergy += nonCoulombEnergy; }
    double getAverageNonCoulombEnergy() const { return _averageNonCoulombEnergy; }
};

#endif // _OUTPUT_DATA_H_
