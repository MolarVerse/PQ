#ifndef _CONFIGURATION_READER_HPP_

#define _CONFIGURATION_READER_HPP_

#include "extxyzReader.hpp"
#include "frame.hpp"
#include "vector3d.hpp"

#include <fstream>
#include <optional>
#include <string>
#include <vector>

class ConfigurationReader
{
  private:
    std::vector<std::string> _filenames;

    std::fstream _fp;

    size_t               _nAtoms  = 0;
    size_t               _nFrames = 0;
    linearAlgebra::Vec3D _box;

    frameTools::Frame _frame;

    [[no_unique_address]] ExtxyzReader _extxyzReader;

  public:
    ConfigurationReader() = default;
    explicit ConfigurationReader(const std::vector<std::string> &filenames);

    bool               nextFrame();
    frameTools::Frame &getFrame();
    void               parseHeader();
    void               parseAtoms();
    [[nodiscard]] bool isBoxSet(const linearAlgebra::Vec3D &box) const { return fabs(box) > 1e-15; }
};

#endif   // _CONFIGURATION_READER_HPP_