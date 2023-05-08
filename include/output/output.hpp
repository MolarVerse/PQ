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
    explicit Output(const std::string &filename) : _filename(filename){};

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
    explicit EnergyOutput(const std::string &filename) : Output(filename){};
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
    explicit TrajectoryOutput(const std::string &filename) : Output(filename){};
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
    explicit LogOutput(const std::string &filename) : Output(filename){};

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
    explicit StdoutOutput(const std::string &filename) : Output(filename){};

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
    explicit RstFileOutput(const std::string &filename) : Output(filename){};
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
    explicit ChargeOutput(const std::string &filename) : Output(filename){};
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
    explicit InfoOutput(const std::string &filename) : Output(filename){};
};

#endif