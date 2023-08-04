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
        line         = StringUtilities::removeComments(line, "#");
        lineElements = StringUtilities::splitString(line);

        if (lineElements.empty())
        {
            ++_lineNumber;
            continue;
        }

        if (StringUtilities::toLowerCopy(lineElements[0]) == "end")
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
 */
void TypesSection::processSection(vector<string> &lineElements, engine::Engine &engine)
{
    if (lineElements.size() != 8)
        throw customException::ParameterFileException("Wrong number of arguments in parameter file types section at line " +
                                                      to_string(_lineNumber) + " - number of elements has to be 8!");

    const auto scaleCoulomb     = stod(lineElements[6]);
    const auto scaleVanDerWaals = stod(lineElements[7]);

    if (scaleCoulomb < 0.0 || scaleCoulomb > 1.0)
        throw customException::ParameterFileException("Wrong scaleCoulomb in parameter file types section at line " +
                                                      to_string(_lineNumber) + " - has to be between 0 and 1!");

    if (scaleVanDerWaals < 0.0 || scaleVanDerWaals > 1.0)
        throw customException::ParameterFileException("Wrong scaleVanDerWaals in parameter file types section at line " +
                                                      to_string(_lineNumber) + " - has to be between 0 and 1!");

    engine.getForceField().setScale14Coulomb(scaleCoulomb);
    engine.getForceField().setScale14VanDerWaals(scaleVanDerWaals);
}

/**
 * @brief processes the bond section of the parameter file
 *
 * @param line
 * @param engine
 */
void BondSection::processSection(vector<string> &lineElements, engine::Engine &engine)
{
    if (lineElements.size() != 3)
        throw customException::ParameterFileException("Wrong number of arguments in parameter file bond section at line " +
                                                      to_string(_lineNumber) + " - number of elements has to be 3!");

    auto id                  = stoul(lineElements[0]);
    auto equilibriumDistance = stod(lineElements[1]);
    auto forceConstant       = stod(lineElements[2]);

    if (equilibriumDistance < 0.0)
        throw customException::ParameterFileException("Parameter file bond section at line " + to_string(_lineNumber) +
                                                      " - equilibrium distance has to be positive!");

    auto bondType = forceField::BondType(id, equilibriumDistance, forceConstant);

    engine.getForceField().addBondType(bondType);
}

/**
 * @brief processes the angle section of the parameter file
 *
 * @param line
 * @param engine
 */
void AngleSection::processSection(vector<string> &lineElements, engine::Engine &engine)
{
    if (lineElements.size() != 3)
        throw customException::ParameterFileException("Wrong number of arguments in parameter file angle section at line " +
                                                      to_string(_lineNumber) + " - number of elements has to be 3!");

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
 */
void DihedralSection::processSection(vector<string> &lineElements, engine::Engine &engine)
{
    if (lineElements.size() != 4)
        throw customException::ParameterFileException("Wrong number of arguments in parameter file dihedral section at line " +
                                                      to_string(_lineNumber) + " - number of elements has to be 4!");

    auto id            = stoul(lineElements[0]);
    auto forceConstant = stod(lineElements[1]);
    auto periodicity   = stod(lineElements[2]);
    auto phaseShift    = stod(lineElements[3]);

    if (periodicity < 0.0)
        throw customException::ParameterFileException("Parameter file dihedral section at line " + to_string(_lineNumber) +
                                                      " - periodicity has to be positive!");

    auto dihedralType = forceField::DihedralType(id, forceConstant, periodicity, phaseShift);

    engine.getForceField().addDihedralType(dihedralType);
}

/**
 * @brief processes the improper section of the parameter file
 *
 * @param line
 * @param engine
 */
void ImproperDihedralSection::processSection(vector<string> &lineElements, engine::Engine &engine)
{
    if (lineElements.size() != 4)
        throw customException::ParameterFileException("Wrong number of arguments in parameter file improper section at line " +
                                                      to_string(_lineNumber) + " - number of elements has to be 4!");

    auto id            = stoul(lineElements[0]);
    auto forceConstant = stod(lineElements[1]);
    auto periodicity   = stod(lineElements[2]);
    auto phaseShift    = stod(lineElements[3]);

    if (periodicity < 0.0)
        throw customException::ParameterFileException("Parameter file improper section at line " + to_string(_lineNumber) +
                                                      " - periodicity has to be positive!");

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
 */
void NonCoulombicsSection::processHeader(vector<string> &lineElements, engine::Engine &engine)
{
    if (lineElements.size() != 1 && lineElements.size() != 2)
        throw customException::ParameterFileException(
            "Wrong number of arguments in parameter file in nonCoulombics header at line " + to_string(_lineNumber) +
            " - number of elements has to be 1 or 2!");

    if (lineElements.size() == 2)
    {

        const auto type = StringUtilities::toLowerCopy(lineElements[1]);

        if (type == "lj")
            engine.getForceField().setNonCoulombicType(forceField::NonCoulombicType::LJ);
        else if (type == "buckingham")
            engine.getForceField().setNonCoulombicType(forceField::NonCoulombicType::BUCKINGHAM);
        else if (type == "morse")
            engine.getForceField().setNonCoulombicType(forceField::NonCoulombicType::MORSE);
        else
            throw customException::ParameterFileException(
                "Wrong type of nonCoulombic in parameter file nonCoulombic section at line " + to_string(_lineNumber) +
                " - has to be lj, buckingham or morse!");
    }

    _nonCoulombicType = engine.getForceField().getNonCoulombicType();
}

/**
 * @brief processes the nonCoulombics section of the parameter file
 *
 * @param line
 * @param engine
 */
void NonCoulombicsSection::processSection(vector<string> &lineElements, engine::Engine &engine)
{
    switch (_nonCoulombicType)
    {
    case forceField::NonCoulombicType::LJ: processLJ(lineElements, engine); break;
    case forceField::NonCoulombicType::BUCKINGHAM: processBuckingham(lineElements, engine); break;
    case forceField::NonCoulombicType::MORSE: processMorse(lineElements, engine); break;
    default:
        throw customException::ParameterFileException(
            "Wrong type of nonCoulombic in parameter file nonCoulombic section at line " + to_string(_lineNumber) +
            " - has to be lj, buckingham or morse!");
    }
}

/**
 * @brief processes the LJ nonCoulombics section of the parameter file
 *
 * @param line
 * @param engine
 */
void NonCoulombicsSection::processLJ(vector<string> &lineElements, engine::Engine &engine) const
{
    if (lineElements.size() != 4 && lineElements.size() != 5)
        throw customException::ParameterFileException(
            "Wrong number of arguments in parameter file in Lennard Jones nonCoulombics section at line " +
            to_string(_lineNumber) + " - number of elements has to be 4 or 5!");

    const auto atomType1 = stoul(lineElements[0]);
    const auto atomType2 = stoul(lineElements[1]);
    const auto c6        = stod(lineElements[2]);
    const auto c12       = stod(lineElements[3]);

    const auto cutOff = lineElements.size() == 5 ? stod(lineElements[4]) : -1.0;

    engine.getForceField().addNonCoulombicPair(make_unique<forceField::LennardJonesPair>(atomType1, atomType2, cutOff, c6, c12));
}

/**
 * @brief processes the Buckingham nonCoulombics section of the parameter file
 *
 * @param line
 * @param engine
 */
void NonCoulombicsSection::processBuckingham(vector<string> &lineElements, engine::Engine &engine) const
{
    if (lineElements.size() != 5 && lineElements.size() != 6)
        throw customException::ParameterFileException(
            "Wrong number of arguments in parameter file in Buckingham nonCoulombics section at line " + to_string(_lineNumber) +
            " - number of elements has to be 5 or 6!");

    const auto atomType1 = stoul(lineElements[0]);
    const auto atomType2 = stoul(lineElements[1]);
    const auto a         = stod(lineElements[2]);
    const auto dRho      = stod(lineElements[3]);
    const auto c6        = stod(lineElements[4]);

    const auto cutOff = lineElements.size() == 6 ? stod(lineElements[5]) : -1.0;

    engine.getForceField().addNonCoulombicPair(
        make_unique<forceField::BuckinghamPair>(atomType1, atomType2, cutOff, a, dRho, c6));
}

/**
 * @brief processes the Morse nonCoulombics section of the parameter file
 *
 * @param line
 * @param engine
 */
void NonCoulombicsSection::processMorse(vector<string> &lineElements, engine::Engine &engine) const
{
    if (lineElements.size() != 5 && lineElements.size() != 6)
        throw customException::ParameterFileException(
            "Wrong number of arguments in parameter file in Morse nonCoulombics section at line " + to_string(_lineNumber) +
            " - number of elements has to be 5 or 6!");

    const auto atomType1           = stoul(lineElements[0]);
    const auto atomType2           = stoul(lineElements[1]);
    const auto dissociationEnergy  = stod(lineElements[2]);
    const auto wellWidth           = stod(lineElements[3]);
    const auto equilibriumDistance = stod(lineElements[4]);

    const auto cutOff = lineElements.size() == 6 ? stod(lineElements[5]) : -1.0;

    engine.getForceField().addNonCoulombicPair(
        make_unique<forceField::MorsePair>(atomType1, atomType2, cutOff, dissociationEnergy, wellWidth, equilibriumDistance));
}