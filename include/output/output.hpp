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
    static int _outputFreq;

    void openFile();

public:
    std::string getFilename() const { return _filename; };
    void setFilename(std::string_view filename);

    static int getOutputFreq() { return _outputFreq; };
    static void setOutputFreq(int outputFreq);
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
    EnergyOutput() = default;
    EnergyOutput &operator=(const EnergyOutput &output) { return *this; }
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
    TrajectoryOutput() = default;
    TrajectoryOutput &operator=(const TrajectoryOutput &) { return *this; }
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
    LogOutput() = default;
    LogOutput &operator=(const LogOutput &) { return *this; }

    void writeDensityWarning();
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
    StdoutOutput() = default;
    StdoutOutput &operator=(const StdoutOutput &) { return *this; }

    void writeDensityWarning() const;
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
    RstFileOutput() = default;
    RstFileOutput &operator=(const RstFileOutput &) { return *this; }
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
    ChargeOutput() = default;
    ChargeOutput &operator=(const ChargeOutput &) { return *this; }
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
    InfoOutput() = default;
    InfoOutput &operator=(const InfoOutput &) { return *this; }
};

#endif