/*****************************************************************************
<GPL_HEADER>

    PQ
    Copyright (C) 2023-now  Jakob Gamper

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

<GPL_HEADER>
******************************************************************************/

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <memory>
#include <vector>

#include "atom.hpp"
#include "berendsenManostat.hpp"
#include "constants/conversionFactors.hpp"
#include "constants/internalConversionFactors.hpp"
#include "enums/manostat.hpp"
#include "exceptions.hpp"
#include "generalSettings.hpp"
#include "manostatSettings.hpp"
#include "molecule.hpp"
#include "orthorhombicBox.hpp"
#include "physicalData.hpp"
#include "potentialSettings.hpp"
#include "resetKinetics.hpp"
#include "resetKineticsSettings.hpp"
#include "settings.hpp"
#include "simulationBox.hpp"
#include "stochasticRescalingManostat.hpp"
#include "thermostatSettings.hpp"
#include "throwWithMessage.hpp"
#include "timingsSettings.hpp"
#include "triclinicBox.hpp"

namespace
{
    class ManostatRegression : public ::testing::Test
    {
       protected:
        molsys::SimulationBox      _box;
        physicalData::PhysicalData _data;
        std::unique_ptr<Settings>  _settings;

        void SetUp() override
        {
            _settings = std::make_unique<Settings>();
            settings::GeneralSettings::setVirialType(VirialType::MOLECULAR);
            settings::GeneralSettings::setRandomSeed(42);
            settings::GeneralSettings::setIsRandomSeedSet(true);
            _settings->manostatSettings.setIsotropy(Isotropy::ISOTROPIC);
            _settings->manostatSettings.setFixedAxis(FixedAxis::NONE);
            _settings->manostatSettings.setTargetPressure(1.0);
            _settings->manostatSettings.setTauManostat(1.0);
            _settings->manostatSettings.setCompressibility(0.03);
            settings::TimingsSettings::setTimeStep(1.0);
            settings::ThermostatSettings::setActualTargetTemperature(0.0);
            settings::PotentialSettings::setCoulombRadiusCutOff(0.1);
            _box.setBox(molsys::OrthorhombicBox({10.0, 10.0, 10.0}), false);
            _data.setVirial(linalg::tensor3D{0.0});
            _data.setKineticEnergyMolecularVector(linalg::tensor3D{0.0});
            _data.setKineticEnergyAtomicVector(linalg::tensor3D{0.0});
        }

        void _addMolecule(
            const std::vector<double>& positions,
            const double               speed
        )
        {
            molsys::Molecule molecule;
            for (const auto x : positions)
            {
                auto atom = std::make_shared<molsys::Atom>();
                atom->setMass(1.0);
                atom->setPosition({x, 0.0, 0.0});
                atom->setVelocity({speed, 0.0, 0.0});
                molecule.addAtom(atom);
                _box.addAtom(atom);
            }
            molecule.calculateCenterOfMass(_box.getBox());
            _box.addMolecule(molecule);
            _box.calculateTotalMass();
            _box.calculateDensity();
            _box.calculateDegreesOfFreedom();
            _data.setVolume(_box.getVolume());
            _data.setDensity(_box.getDensity());
        }
    };
}   // namespace

TEST_F(ManostatRegression, triclinicHydrostaticBalancePreservesCell)
{
    _box.setBox(
        molsys::TriclinicBox({10.0, 10.0, 10.0}, {90.0, 90.0, 60.0}, true),
        false
    );
    _addMolecule({0.0}, 0.0);
    _addMolecule({1.0}, 0.0);
    _data.setVirial(linalg::diagonalMatrix(_box.getVolume() / PRESSURE_FACTOR));
    _settings->manostatSettings.setIsotropy(Isotropy::FULL_ANISOTROPIC);
    manostat::FullAnisotropicBerendsenManostat berendsen(
        _settings->manostatSettings
    );
    manostat::FullAnisotropicStochasticRescalingManostat stochastic(
        _settings->manostatSettings
    );
    for (auto* coupling :
         std::vector<manostat::Manostat*>{&berendsen, &stochastic})
    {
        const auto oldCell =
            _box.getBox().getBoxMatrix(molsys::Periodicity::XYZ);
        coupling->applyManostat(_box, _data);
        EXPECT_NEAR(_data.getPressure(), 1.0, 1e-12);
        for (size_t i = 0; i < 3; ++i)
        {
            for (size_t j = 0; j < 3; ++j)
            {
                EXPECT_NEAR(
                    _box.getBox().getBoxMatrix(molsys::Periodicity::XYZ)[i][j],
                    oldCell[i][j],
                    1e-12
                );
            }
        }
    }
}

TEST_F(ManostatRegression, triclinicLengthPressureMatchesMolecularWork)
{
    _box.setBox(
        molsys::TriclinicBox({10.0, 10.0, 10.0}, {90.0, 90.0, 60.0}, false),
        false
    );
    _settings->manostatSettings.setFixedAxis(FixedAxis::YZ);
    _data.setVirial(
        linalg::tensor3D{{2.0, -1.0, 0.5}, {4.0, -2.0, 1.0}, {6.0, -3.0, 1.5}}
    );
    manostat::Manostat pressure(_settings->manostatSettings);
    pressure.calculatePressure(_box, _data);
    EXPECT_NEAR(
        _data.getPressure(),
        0.5 * PRESSURE_FACTOR / _box.getVolume(),
        1e-12
    );
    // r=(1,2,3), F=(2,-1,0.5): changing the first cell length
    // moves r_x by (1 - 2/sqrt(3)) times the fractional length change.
    EXPECT_NEAR(
        _data.getCoupledPressure(),
        (2.0 - 4.0 / std::sqrt(3.0)) * PRESSURE_FACTOR / _box.getVolume(),
        1e-12
    );
}

TEST_F(ManostatRegression, berendsenModesHaveSameHydrostaticVolumeResponse)
{
    constexpr double increment = 1e-6;
    for (const auto fixed : {FixedAxis::NONE, FixedAxis::Z})
    {
        _settings->manostatSettings.setFixedAxis(fixed);
        _settings->manostatSettings.setCompressibility(increment);

        // Each mode needs its own settings object: the manostats below hold
        // a reference, not a copy, so sharing one object would make every
        // mode see only the isotropy set last.
        auto isoSettings = _settings->manostatSettings;
        isoSettings.setIsotropy(Isotropy::ISOTROPIC);
        manostat::BerendsenManostat iso(isoSettings);

        auto semiSettings = _settings->manostatSettings;
        semiSettings.setIsotropy(Isotropy::SEMI_ISOTROPIC_XY);
        manostat::SemiIsotropicBerendsenManostat semi(semiSettings);

        auto anisoSettings = _settings->manostatSettings;
        anisoSettings.setIsotropy(Isotropy::ANISOTROPIC);
        manostat::AnisotropicBerendsenManostat aniso(anisoSettings);

        auto fullSettings = _settings->manostatSettings;
        fullSettings.setIsotropy(Isotropy::FULL_ANISOTROPIC);
        manostat::FullAnisotropicBerendsenManostat full(fullSettings);

        for (auto* coupling : std::vector<manostat::BerendsenManostat*>{
                 &iso,
                 &semi,
                 &aniso,
                 &full
             })
        {
            coupling->calculatePressure(_box, _data);
            const auto mu = coupling->calculateMu();
            EXPECT_NEAR((1.0 - det(mu)) / increment, 1.0, 1e-6);
            if (fixed == FixedAxis::Z)
            {
                EXPECT_DOUBLE_EQ(mu[2][2], 1.0);
            }
        }
    }
}

TEST_F(ManostatRegression, molecularCouplingAllowsAtomicVirial)
{
    _addMolecule({-0.2, 0.2}, 0.0);
    settings::GeneralSettings::setVirialType(VirialType::ATOMIC);
    manostat::BerendsenManostat berendsen(_settings->manostatSettings);
    manostat::StochasticRescalingManostat stochastic(_settings->manostatSettings
    );
    testing::internal::CaptureStdout();
    for (size_t step = 0; step < 2; ++step)
    {
        for (auto* coupling :
             std::vector<manostat::Manostat*>{&berendsen, &stochastic})
        {
            const auto oldVolume = _box.getVolume();
            EXPECT_NO_THROW(coupling->applyManostat(_box, _data));
            EXPECT_LT(_box.getVolume(), oldVolume);
            EXPECT_NEAR(
                _box.getAtom(1).getPosition()[0] -
                    _box.getAtom(0).getPosition()[0],
                0.4,
                1e-12
            );
        }
    }
    EXPECT_TRUE(testing::internal::GetCapturedStdout().empty());
    manostat::Manostat reportPressure(_settings->manostatSettings);
    EXPECT_NO_THROW(reportPressure.applyManostat(_box, _data));
}

TEST_F(ManostatRegression, invalidScalingPreservesSimulationState)
{
    // Each proposal needs its own settings object: the manostats below hold
    // a reference, not a copy, so sharing one object would make every
    // proposal see only the values set last (e.g. overflow/underflow would
    // both end up with cutoff's target pressure and compressibility).
    _settings->manostatSettings.setFixedAxis(FixedAxis::NONE);
    _settings->manostatSettings.setTargetPressure(1.0);
    _settings->manostatSettings.setTauManostat(1.0);

    auto negativeRootSettings = _settings->manostatSettings;
    negativeRootSettings.setIsotropy(Isotropy::SEMI_ISOTROPIC_XY);
    negativeRootSettings.setCompressibility(2.0);
    manostat::SemiIsotropicBerendsenManostat negativeRoot(negativeRootSettings);

    auto negativeVolumeSettings = _settings->manostatSettings;
    negativeVolumeSettings.setIsotropy(Isotropy::ISOTROPIC);
    negativeVolumeSettings.setCompressibility(2.0);
    manostat::BerendsenManostat negativeVolume(negativeVolumeSettings);

    auto overflowSettings = negativeVolumeSettings;
    overflowSettings.setTargetPressure(-3000.0);
    overflowSettings.setCompressibility(1.0);
    manostat::StochasticRescalingManostat overflow(overflowSettings);

    auto underflowSettings = negativeVolumeSettings;
    underflowSettings.setTargetPressure(3000.0);
    underflowSettings.setCompressibility(1.0);
    manostat::StochasticRescalingManostat underflow(underflowSettings);

    auto cutoffSettings = negativeVolumeSettings;
    cutoffSettings.setCompressibility(0.875);
    manostat::BerendsenManostat cutoff(cutoffSettings);

    const auto proposals = std::array<manostat::Manostat*, 5>{
        &negativeRoot,
        &negativeVolume,
        &overflow,
        &underflow,
        &cutoff
    };
    const auto proposalSettings = std::array<ManostatSettings*, 5>{
        &negativeRootSettings,
        &negativeVolumeSettings,
        &overflowSettings,
        &underflowSettings,
        &cutoffSettings
    };
    for (size_t i = 0; i < proposals.size(); ++i)
    {
        SCOPED_TRACE(i);
        _box = molsys::SimulationBox{};
        _box.setBox(molsys::OrthorhombicBox({10.0, 10.0, 10.0}), false);
        _addMolecule({4.95, -4.85}, 1.0);
        const auto oldCenter  = _box.getMolecule(0).getCenterOfMass();
        const auto oldDensity = _box.getDensity();
        proposalSettings.at(i)->setIsotropy(
            i == 0 ? Isotropy::SEMI_ISOTROPIC_XY : Isotropy::ISOTROPIC
        );
        settings::PotentialSettings::setCoulombRadiusCutOff(i == 4 ? 3.0 : 0.1);
        EXPECT_THROW_MSG(
            proposals.at(i)->applyManostat(_box, _data),
            exc::ManostatException,
            i == 4
                ? "Coulomb radius cut off is larger than half of the minimal "
                  "box dimension"
                : "Invalid manostat scaling matrix"
        );
        EXPECT_EQ(_box.getBoxDimensions(), linalg::Vec3D(10.0));
        EXPECT_DOUBLE_EQ(_box.getVolume(), 1000.0);
        EXPECT_DOUBLE_EQ(_box.getDensity(), oldDensity);
        EXPECT_DOUBLE_EQ(_data.getVolume(), 1000.0);
        EXPECT_DOUBLE_EQ(_data.getDensity(), oldDensity);
        EXPECT_EQ(_box.getMolecule(0).getCenterOfMass(), oldCenter);
        EXPECT_DOUBLE_EQ(_box.getAtom(0).getPosition()[0], 4.95);
        EXPECT_DOUBLE_EQ(_box.getAtom(1).getPosition()[0], -4.85);
        EXPECT_EQ(_box.getAtom(0).getVelocity(), linalg::Vec3D(1.0, 0.0, 0.0));
    }
}

namespace
{
    template <size_t N>
    struct NoiseMoments
    {
        size_t                               count = 0;
        std::array<double, N>                sums{};
        std::array<std::array<double, N>, N> products{};

        void add(const std::array<double, N>& sample)
        {
            ++count;
            for (size_t i = 0; i < N; ++i)
            {
                sums.at(i) += sample.at(i);
                for (size_t j = 0; j < N; ++j)
                {
                    products.at(i).at(j) += sample.at(i) * sample.at(j);
                }
            }
        }

        void expectIndependentStandardNormals() const
        {
            const auto sampleCount = static_cast<double>(count);
            for (size_t i = 0; i < N; ++i)
            {
                const auto mean = sums.at(i) / sampleCount;
                EXPECT_NEAR(mean, 0.0, 0.06);
                for (size_t j = 0; j < N; ++j)
                {
                    const auto covariance =
                        (products.at(i).at(j) / sampleCount) -
                        (mean * sums.at(j) / sampleCount);
                    EXPECT_NEAR(covariance, i == j ? 1.0 : 0.0, 0.08);
                }
            }
        }
    };
}   // namespace

TEST_F(ManostatRegression, stochasticCellModesHaveIndependentNoise)
{
    constexpr double temperature     = 300.0;
    constexpr double tau             = 1e8;
    constexpr double compressibility = 4.591e-5;
    settings::ThermostatSettings::setActualTargetTemperature(temperature);

    ManostatSettings semiSettings;
    semiSettings.setTargetPressure(0.0);
    semiSettings.setTauManostat(tau);
    semiSettings.setCompressibility(compressibility);
    semiSettings.setIsotropy(Isotropy::SEMI_ISOTROPIC_XY);
    semiSettings.setFixedAxis(FixedAxis::NONE);
    manostat::SemiIsotropicStochasticRescalingManostat semi(semiSettings);

    ManostatSettings freeSettings;
    freeSettings.setTargetPressure(0.0);
    freeSettings.setTauManostat(tau);
    freeSettings.setCompressibility(compressibility);
    freeSettings.setFixedAxis(FixedAxis::NONE);
    manostat::AnisotropicStochasticRescalingManostat     aniso(freeSettings);
    manostat::FullAnisotropicStochasticRescalingManostat full(freeSettings);

    ManostatSettings fixedSettings;
    fixedSettings.setTargetPressure(0.0);
    fixedSettings.setTauManostat(tau);
    fixedSettings.setCompressibility(compressibility);
    fixedSettings.setFixedAxis(FixedAxis::Y);
    manostat::FullAnisotropicStochasticRescalingManostat fixed(fixedSettings);

    for (auto* coupling :
         std::vector<manostat::Manostat*>{&semi, &aniso, &full, &fixed})
        coupling->calculatePressure(_box, _data);

    const auto noiseFactor = BOLTZMANN_CONSTANT_IN_KCAL_PER_MOL * temperature *
                             compressibility * PRESSURE_FACTOR /
                             (3.0 * tau * _box.getVolume());
    const auto      lengthSigma = std::sqrt(2.0 * noiseFactor);
    const auto      shearSigma  = std::sqrt(4.0 * noiseFactor);
    const auto      volumeSigma = std::sqrt(6.0 * noiseFactor);
    NoiseMoments<2> areaHeight;
    NoiseMoments<3> lengths;
    NoiseMoments<3> shape;
    NoiseMoments<3> fixedShape;
    NoiseMoments<1> semiVolume;
    NoiseMoments<1> anisoVolume;
    NoiseMoments<1> fullVolume;
    for (size_t sample = 0; sample < 8192; ++sample)
    {
        const auto semiMu  = semi.calculateMu(_box.getVolume());
        const auto anisoMu = aniso.calculateMu(_box.getVolume());
        const auto fullMu  = full.calculateMu(_box.getVolume());
        const auto fixedMu = fixed.calculateMu(_box.getVolume());
        areaHeight.add(
            {std::log(semiMu[0][0] * semiMu[1][1]) /
                 (2.0 * std::sqrt(noiseFactor)),
             std::log(semiMu[2][2]) / lengthSigma}
        );
        lengths.add(
            {std::log(anisoMu[0][0]) / lengthSigma,
             std::log(anisoMu[1][1]) / lengthSigma,
             std::log(anisoMu[2][2]) / lengthSigma}
        );
        shape.add(
            {(fullMu[0][0] - 1.0) / lengthSigma,
             (fullMu[1][1] - 1.0) / lengthSigma,
             fullMu[0][1] / shearSigma}
        );
        fixedShape.add(
            {(fixedMu[0][0] - 1.0) / lengthSigma,
             (fixedMu[2][2] - 1.0) / lengthSigma,
             fixedMu[0][2] / shearSigma}
        );
        semiVolume.add({std::log(det(semiMu)) / volumeSigma});
        anisoVolume.add({std::log(det(anisoMu)) / volumeSigma});
        fullVolume.add({std::log(det(fullMu)) / volumeSigma});
    }
    areaHeight.expectIndependentStandardNormals();
    lengths.expectIndependentStandardNormals();
    shape.expectIndependentStandardNormals();
    fixedShape.expectIndependentStandardNormals();
    semiVolume.expectIndependentStandardNormals();
    anisoVolume.expectIndependentStandardNormals();
    fullVolume.expectIndependentStandardNormals();
}

TEST_F(ManostatRegression, stochasticRescalingRefreshesKineticsBeforeReset)
{
    constexpr double speed = 1e13;
    constexpr double scale = 0.98;
    _addMolecule({-1.0}, speed);
    _addMolecule({1.0}, speed);
    _data.calculateKinetics(_box);
    _data.calculateTemperature(_box);
    const auto expectedEnergy = _data.getKineticEnergy() / (scale * scale);
    manostat::Manostat pressure(_settings->manostatSettings);
    pressure.calculatePressure(_box, _data);

    _settings->manostatSettings.setTargetPressure(
        _data.getPressure() - (3.0 * std::log(scale))
    );
    _settings->manostatSettings.setTauManostat(1.0);
    _settings->manostatSettings.setCompressibility(1.0);
    _settings->manostatSettings.setFixedAxis(FixedAxis::NONE);
    manostat::StochasticRescalingManostat stochastic(_settings->manostatSettings
    );
    stochastic.applyManostat(_box, _data);

    EXPECT_NEAR(_data.getKineticEnergy() / expectedEnergy, 1.0, 1e-12);
    EXPECT_NEAR(
        _data.getTemperature() / _box.calculateTemperature(),
        1.0,
        1e-12
    );
    EXPECT_NEAR(_data.getMomentum()[0] / (speed * FS_TO_S), 2.0 / scale, 1e-12);
    EXPECT_NEAR(
        _data.getKinEnergyMolTensor()[0][0] / expectedEnergy,
        1.0,
        1e-12
    );
    EXPECT_NEAR(
        _data.getKinEnergyAtomTensor()[0][0] / expectedEnergy,
        1.0,
        1e-12
    );

    ResetKineticsSettings resetSettings;
    resetSettings.setFScale(100);
    resetSettings.setFReset(1);
    resetSettings.setFResetAngular(100);
    resetKinetics::ResetKinetics reset(resetSettings);
    reset.reset(1, _data, _box);
    EXPECT_NEAR(_box.calculateMomentum()[0] / speed, 0.0, 1e-12);
}

TEST_F(ManostatRegression, atomicVirialSupportsMonatomicPressureCoupling)
{
    _addMolecule({-1.0}, 1e13);
    _addMolecule({1.0}, -1e13);
    _data.calculateKinetics(_box);
    settings::GeneralSettings::setVirialType(VirialType::ATOMIC);
    _settings->manostatSettings.setCompressibility(0.001);
    manostat::BerendsenManostat berendsen(_settings->manostatSettings);
    manostat::StochasticRescalingManostat stochastic(_settings->manostatSettings
    );
    EXPECT_NO_THROW(berendsen.applyManostat(_box, _data));
    EXPECT_NO_THROW(stochastic.applyManostat(_box, _data));
    EXPECT_TRUE(std::isfinite(_box.getVolume()));
    EXPECT_GT(_box.getVolume(), 0.0);
}
