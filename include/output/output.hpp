#ifndef _OUTPUT_H_

#define _OUTPUT_H_

#include <string>
#include <fstream>

/**
 * @class Output
 *
 * @brief Base class for output files
 *
 */
class Output
{
protected:
    std::string _filename;
    std::ofstream _fp;
    inline static int _outputFreq = 1;

    void openFile();

public:
    explicit Output(const std::string &filename) : _filename(filename){};

    void setFilename(std::string_view filename);

    static void setOutputFreq(int outputFreq);

    std::string initialMomentumMessage(double momentum) const;

    // standard getter and setters
    std::string getFilename() const { return _filename; };
    static int getOutputFreq() { return _outputFreq; };
};

/**
 * @class EnergyOutput inherits from Output
 *
 * @brief Output file for energy
 *
 */
class EnergyOutput : public Output
{
public:
    using Output::Output;
};

/**
 * @class TrajectoryOutput inherits from Output
 *
 * @brief Output file for xyz, vel, force files
 *
 */
class TrajectoryOutput : public Output
{
public:
    using Output::Output;
};

/**
 * @class LogOutput inherits from Output
 *
 * @brief Output file for log file
 *
 */
class LogOutput : public Output
{
public:
    using Output::Output;

    void writeDensityWarning();
    void writeInitialMomentum(double momentum);
};

/**
 * @class StdoutOutput inherits from Output
 *
 * @brief Output file for stdout
 *
 */
class StdoutOutput : public Output
{
public:
    using Output::Output;

    void writeDensityWarning() const;
    void writeInitialMomentum(double momentum) const;
};

/**
 * @class RstFileOutput inherits from Output
 *
 * @brief Output file for restart file
 *
 */
class RstFileOutput : public Output
{
public:
    using Output::Output;
};

/**
 * @class ChargeOutput inherits from Output
 *
 * @brief Output file for charge file
 *
 */
class ChargeOutput : public Output
{
public:
    using Output::Output;
};

/**
 * @class InfoOutput inherits from Output
 *
 * @brief Output file for info file
 *
 */
class InfoOutput : public Output
{
public:
    using Output::Output;
};

#endif