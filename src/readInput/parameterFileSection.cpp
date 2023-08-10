#include "parameterFileSection.hpp"

#include "bondType.hpp"
#include "exceptions.hpp"
#include "stringUtilities.hpp"

using namespace std;
using namespace readInput::parameterFile;

/**
 * @brief general process function for parameter sections
 *
 * @param line
 * @param engine
 */
void ParameterFileSection::process(vector<string> &lineElements, engine::Engine &engine)
{
    processHeader(lineElements, engine);

    string line;
    auto   endedNormal = false;

    while (getline(*_fp, line))
    {
        line         = utilities::removeComments(line, "#");
        lineElements = utilities::splitString(line);

        if (lineElements.empty())
        {
            ++_lineNumber;
            continue;
        }

        if (utilities::toLowerCopy(lineElements[0]) == "end")
        {
            ++_lineNumber;
            endedNormal = true;
            break;
        }

        processSection(lineElements, engine);

        ++_lineNumber;
    }

    endedNormally(endedNormal);
}

/**
 * @brief check if section ended normally
 *
 * @param endedNormally
 *
 * @throw customException::ParameterFileException if section did not end normally
 */
void ParameterFileSection::endedNormally(bool endedNormally)
{
    if (!endedNormally)
        throw customException::ParameterFileException("Parameter file " + keyword() + " section ended abnormally!");
}

/**
 * @brief dummy function to forward process of one line to processSection
 *
 * @param lineElements
 * @param engine
 */
void TypesSection::process(vector<string> &lineElements, engine::Engine &engine) { processSection(lineElements, engine); }

/**
 * @brief process types section
 *
 * @param lineElements
 * @param engine
 *
 * @throw customException::ParameterFileException if number of elements in line is not 8
 * @throw customException::ParameterFileException if scaleCoulomb is not between 0 and 1
 * @throw customException::ParameterFileException if scaleVanDerWaals is not between 0 and 1
 */
void TypesSection::processSection(vector<string> &lineElements, engine::Engine &engine)
{
    if (lineElements.size() != 8)
        throw customException::ParameterFileException(
            format("Wrong number of arguments in parameter file types section at line {} - number of elements has to be 8!",
                   _lineNumber));

    const auto scaleCoulomb     = stod(lineElements[6]);
    const auto scaleVanDerWaals = stod(lineElements[7]);

    if (scaleCoulomb < 0.0 || scaleCoulomb > 1.0)
        throw customException::ParameterFileException(
            format("Wrong scaleCoulomb in parameter file types section at line {} - has to be between 0 and 1!", _lineNumber));

    if (scaleVanDerWaals < 0.0 || scaleVanDerWaals > 1.0)
        throw customException::ParameterFileException(format(
            "Wrong scaleVanDerWaals in parameter file types section at line {} - has to be between 0 and 1!", _lineNumber));

    engine.getForceField().setScale14Coulomb(scaleCoulomb);
    engine.getForceField().setScale14VanDerWaals(scaleVanDerWaals);
}

/**
 * @brief processes the bond section of the parameter file
 *
 * @param line
 * @param engine
 *
 * @throw customException::ParameterFileException if number of elements in line is not 3
 * @throw customException::ParameterFileException if equilibrium distance is negative
 */
void BondSection::processSection(vector<string> &lineElements, engine::Engine &engine)
{
    if (lineElements.size() != 3)
        throw customException::ParameterFileException(
            format("Wrong number of arguments in parameter file bond section at line {} - number of elements has to be 3!",
                   _lineNumber));

    auto id                  = stoul(lineElements[0]);
    auto equilibriumDistance = stod(lineElements[1]);
    auto forceConstant       = stod(lineElements[2]);

    if (equilibriumDistance < 0.0)
        throw customException::ParameterFileException(
            format("Parameter file bond section at line {} - equilibrium distance has to be positive!", _lineNumber));

    auto bondType = forceField::BondType(id, equilibriumDistance, forceConstant);

    engine.getForceField().addBondType(bondType);
}

/**
 * @brief processes the angle section of the parameter file
 *
 * @param line
 * @param engine
 *
 * @throw customException::ParameterFileException if number of elements in line is not 3
 */
void AngleSection::processSection(vector<string> &lineElements, engine::Engine &engine)
{
    if (lineElements.size() != 3)
        throw customException::ParameterFileException(
            format("Wrong number of arguments in parameter file angle section at line {} - number of elements has to be 3!",
                   _lineNumber));

    auto id               = stoul(lineElements[0]);
    auto equilibriumAngle = stod(lineElements[1]);
    auto forceConstant    = stod(lineElements[2]);

    auto angleType = forceField::AngleType(id, equilibriumAngle, forceConstant);

    engine.getForceField().addAngleType(angleType);
}

/**
 * @brief processes the dihedral section of the parameter file
 *
 * @param line
 * @param engine
 *
 * @throw customException::ParameterFileException if number of elements in line is not 4
 * @throw customException::ParameterFileException if periodicity is negative
 */
void DihedralSection::processSection(vector<string> &lineElements, engine::Engine &engine)
{
    if (lineElements.size() != 4)
        throw customException::ParameterFileException(
            format("Wrong number of arguments in parameter file angle section at line {} - number of elements has to be 4!",
                   _lineNumber));

    auto id            = stoul(lineElements[0]);
    auto forceConstant = stod(lineElements[1]);
    auto periodicity   = stod(lineElements[2]);
    auto phaseShift    = stod(lineElements[3]);

    if (periodicity < 0.0)
        throw customException::ParameterFileException(
            format("Parameter file dihedral section at line {} - periodicity has to be positive!", _lineNumber));

    auto dihedralType = forceField::DihedralType(id, forceConstant, periodicity, phaseShift);

    engine.getForceField().addDihedralType(dihedralType);
}

/**
 * @brief processes the improper section of the parameter file
 *
 * @param line
 * @param engine
 *
 * @throw customException::ParameterFileException if number of elements in line is not 4
 * @throw customException::ParameterFileException if periodicity is negative
 */
void ImproperDihedralSection::processSection(vector<string> &lineElements, engine::Engine &engine)
{
    if (lineElements.size() != 4)
        throw customException::ParameterFileException(
            format("Wrong number of arguments in parameter file angle section at line {} - number of elements has to be 4!",
                   _lineNumber));

    auto id            = stoul(lineElements[0]);
    auto forceConstant = stod(lineElements[1]);
    auto periodicity   = stod(lineElements[2]);
    auto phaseShift    = stod(lineElements[3]);

    if (periodicity < 0.0)
        throw customException::ParameterFileException(
            format("Parameter file improper section at line {} - periodicity has to be positive!", _lineNumber));

    auto improperType = forceField::DihedralType(id, forceConstant, periodicity, phaseShift);

    engine.getForceField().addImproperDihedralType(improperType);
}

/**
 * @brief processes the nonCoulombics header of the parameter file
 *
 * @note type of forceField can be given as second argument
 *       default is lj (Lennard Jones)
 *
 * @param line
 * @param engine
 *
 * @throw customException::ParameterFileException if number of elements in line is not 1 or 2
 * @throw customException::ParameterFileException if type of nonCoulombic is not lj, buckingham or morse
 */
void NonCoulombicsSection::processHeader(vector<string> &lineElements, engine::Engine &engine)
{
    if (lineElements.size() != 1 && lineElements.size() != 2)
        throw customException::ParameterFileException(format("Wrong number of arguments in parameter file in nonCoulombics "
                                                             "header at line {} - number of elements has to be 1 or 2!",
                                                             _lineNumber));

    if (2 == lineElements.size())
    {
        const auto type = utilities::toLowerCopy(lineElements[1]);

        if (type == "lj")
            engine.getForceField().setNonCoulombicType(forceField::NonCoulombicType::LJ);
        else if (type == "buckingham")
            engine.getForceField().setNonCoulombicType(forceField::NonCoulombicType::BUCKINGHAM);
        else if (type == "morse")
            engine.getForceField().setNonCoulombicType(forceField::NonCoulombicType::MORSE);
        else
            throw customException::ParameterFileException(format("Invalid type of nonCoulombic in parameter file nonCoulombic "
                                                                 "section at line {} - has to be lj, buckingham or morse!",
                                                                 _lineNumber));
    }

    _nonCoulombicType = engine.getForceField().getNonCoulombicType();
}

/**
 * @brief processes the nonCoulombics section of the parameter file
 *
 * @param line
 * @param engine
 *
 * @throw customException::ParameterFileException if nonCoulombic type is not lj, buckingham or morse
 */
void NonCoulombicsSection::processSection(vector<string> &lineElements, engine::Engine &engine)
{
    switch (_nonCoulombicType)
    {
    case forceField::NonCoulombicType::LJ: processLJ(lineElements, engine); break;
    case forceField::NonCoulombicType::BUCKINGHAM: processBuckingham(lineElements, engine); break;
    case forceField::NonCoulombicType::MORSE: processMorse(lineElements, engine); break;
    default:
        throw customException::ParameterFileException(format(
            "Wrong type of nonCoulombic in parameter file nonCoulombic section at line {}  - has to be lj, buckingham or morse!",
            _lineNumber));
    }
}

/**
 * @brief processes the LJ nonCoulombics section of the parameter file
 *
 * @param line
 * @param engine
 *
 * @throw customException::ParameterFileException if number of elements in line is not 4 or 5
 */
void NonCoulombicsSection::processLJ(vector<string> &lineElements, engine::Engine &engine) const
{
    if (lineElements.size() != 4 && lineElements.size() != 5)
        throw customException::ParameterFileException(
            format("Wrong number of arguments in parameter file in Lennard Jones nonCoulombics section at line {} - number of "
                   "elements has to be 4 or 5!",
                   _lineNumber));

    const auto atomType1 = stoul(lineElements[0]);
    const auto atomType2 = stoul(lineElements[1]);
    const auto c6        = stod(lineElements[2]);
    const auto c12       = stod(lineElements[3]);

    const auto cutOff = 5 == lineElements.size() ? stod(lineElements[4]) : -1.0;

    engine.getForceField().addNonCoulombicPair(make_unique<forceField::LennardJonesPair>(atomType1, atomType2, cutOff, c6, c12));
}

/**
 * @brief processes the Buckingham nonCoulombics section of the parameter file
 *
 * @param line
 * @param engine
 *
 * @throw customException::ParameterFileException if number of elements in line is not 5 or 6
 */
void NonCoulombicsSection::processBuckingham(vector<string> &lineElements, engine::Engine &engine) const
{
    if (lineElements.size() != 5 && lineElements.size() != 6)
        throw customException::ParameterFileException(
            format("Wrong number of arguments in parameter file in Lennard Jones nonCoulombics section at line {} - number of "
                   "elements has to be 5 or 6!",
                   _lineNumber));

    const auto atomType1 = stoul(lineElements[0]);
    const auto atomType2 = stoul(lineElements[1]);
    const auto a         = stod(lineElements[2]);
    const auto dRho      = stod(lineElements[3]);
    const auto c6        = stod(lineElements[4]);

    const auto cutOff = 6 == lineElements.size() ? stod(lineElements[5]) : -1.0;

    engine.getForceField().addNonCoulombicPair(
        make_unique<forceField::BuckinghamPair>(atomType1, atomType2, cutOff, a, dRho, c6));
}

/**
 * @brief processes the Morse nonCoulombics section of the parameter file
 *
 * @param line
 * @param engine
 *
 * @throw customException::ParameterFileException if number of elements in line is not 5 or 6
 */
void NonCoulombicsSection::processMorse(vector<string> &lineElements, engine::Engine &engine) const
{
    if (lineElements.size() != 5 && lineElements.size() != 6)
        throw customException::ParameterFileException(
            format("Wrong number of arguments in parameter file in Lennard Jones nonCoulombics section at line {} - number of "
                   "elements has to be 5 or 6!",
                   _lineNumber));

    const auto atomType1           = stoul(lineElements[0]);
    const auto atomType2           = stoul(lineElements[1]);
    const auto dissociationEnergy  = stod(lineElements[2]);
    const auto wellWidth           = stod(lineElements[3]);
    const auto equilibriumDistance = stod(lineElements[4]);

    const auto cutOff = 6 == lineElements.size() ? stod(lineElements[5]) : -1.0;

    engine.getForceField().addNonCoulombicPair(
        make_unique<forceField::MorsePair>(atomType1, atomType2, cutOff, dissociationEnergy, wellWidth, equilibriumDistance));
}