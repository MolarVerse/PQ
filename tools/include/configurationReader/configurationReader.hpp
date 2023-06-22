#ifndef _CONFIGURATIONREADER_HPP_

#define _CONFIGURATIONREADER_HPP_

#include <vector>
#include <string>
#include <fstream>
#include <optional>

#include "vector3d.hpp"
#include "extxyzReader.hpp"
#include "frame.hpp"

class ConfigurationReader
{
private:
    std::vector<std::string> _filenames;

    std::fstream _fp;

    size_t _nAtoms = 0;
    size_t _nFrames = 0;
    Vec3D _box;

    frameTools::Frame _frame;

    ExtxyzReader _extxyzReader;

public:
    ConfigurationReader() = default;
    explicit ConfigurationReader(const std::vector<std::string> &filenames);

    bool nextFrame();
    frameTools::Frame &getFrame();
    void parseHeader();
    void parseAtoms();
    [[nodiscard]] bool isBoxSet(const Vec3D &box) const { return fabs(box) > 1e-15; }
};

#endif // _CONFIGURATIONREADER_HPP_