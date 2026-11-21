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
#include "enums/manostat.hpp"
#include "exceptions.hpp"
#include "generalSettings.hpp"
#include "manostatSettings.hpp"
#include "molecule.hpp"
#include "orthorhombicBox.hpp"
#include "physicalData.hpp"
#include "resetKinetics.hpp"
#include "resetKineticsSettings.hpp"
#include "settings.hpp"
#include "simulationBox.hpp"
#include "stochasticRescalingManostat.hpp"
#include "throwWithMessage.hpp"
#include "triclinicBox.hpp"
#include "unit_types.hpp"
#include "units.hpp"
#include "vector3d/vector3dClass.hpp"

namespace
{
    // pressure in bar corresponding to 1 kcal/mol/A^3
    const double PRESSURE_FACTOR =
        (1.0 * kcal_per_mol / angstrom3 / avogadro_constant)
            .numerical_value_in(bar);
}   // namespace

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
            GeneralSettings::setVirialType(VirialType::MOLECULAR);
            GeneralSettings::setRandomSeed(42);
            GeneralSettings::setIsRandomSeedSet(true);
            _settings->manostatSettings.setIsotropyByUser(Isotropy::ISOTROPIC);
            _settings->manostatSettings.setFixedAxisByUser(FixedAxis::NONE);
            _settings->manostatSettings.setTargetPressureByUser(1.0 * bar);
            _settings->manostatSettings.setTauByUser(1.0 * fs);
            _settings->manostatSettings.setCompressibilityByUser(
                0.03 * per_bar
            );
            _box.setBox(
                molsys::OrthorhombicBox(
                    {10.0 * angstrom, 10.0 * angstrom, 10.0 * angstrom}
                ),
                false
            );
            _data.set_virial(VirialPerMol::zero());
            _data.setKineticEnergyMolecularVector(VirialPerMol::zero());
            _data.setKineticEnergyAtomicVector(VirialPerMol::zero());
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
                atom->setMass(1.0 * g_per_mol);
                atom->setPosition(linalg::Vec3D{x, 0.0, 0.0} * angstrom);
                atom->setVelocity(
                    linalg::Vec3D{speed, 0.0, 0.0} * angstrom / s
                );
                molecule.addAtom(atom);
                _box.addAtom(atom);
            }
            molecule.calculateCenterOfMass(_box.getBox());
            _box.addMolecule(molecule);
            _box.calculateTotalMass();
            _box.calculateDensity();
            _box.calculateDegreesOfFreedom();
            _data.set_volume(_box.getVolume());
            _data.set_density(_box.getDensity());
        }
    };
}   // namespace

TEST_F(ManostatRegression, triclinicHydrostaticBalancePreservesCell)
{
    _box.setBox(
        molsys::TriclinicBox(
            {10.0 * angstrom, 10.0 * angstrom, 10.0 * angstrom},
            {90.0 * deg, 90.0 * deg, 60.0 * deg},
            true
        ),
        false
    );
    _addMolecule({0.0}, 0.0);
    _addMolecule({1.0}, 0.0);
    _data.set_virial(
        linalg::diagonalMatrix(
            _box.getVolume().numerical_value_in(angstrom3) / PRESSURE_FACTOR *
            kcal_per_mol
        )
    );
    _settings->manostatSettings.setIsotropyByUser(Isotropy::FULL_ANISOTROPIC);
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
        const auto timeStep = 1.0 * fs;
        coupling->applyManostat(
            _box,
            _data,
            toTemperature(0.0),
            timeStep,
            0.1 * angstrom
        );
        EXPECT_NEAR(_data.get_pressure().numerical_value_in(bar), 1.0, 1e-12);
        for (size_t i = 0; i < 3; ++i)
        {
            for (size_t j = 0; j < 3; ++j)
            {
                EXPECT_NEAR(
                    _box.getBox()
                        .getBoxMatrix(molsys::Periodicity::XYZ)[i][j]
                        .numerical_value_in(angstrom),
                    oldCell[i][j].numerical_value_in(angstrom),
                    1e-12
                );
            }
        }
    }
}

TEST_F(ManostatRegression, triclinicLengthPressureMatchesMolecularWork)
{
    _box.setBox(
        molsys::TriclinicBox(
            {10.0 * angstrom, 10.0 * angstrom, 10.0 * angstrom},
            {90.0 * deg, 90.0 * deg, 60.0 * deg},
            false
        ),
        false
    );
    _settings->manostatSettings.setFixedAxisByUser(FixedAxis::YZ);
    _data.set_virial(
        linalg::tensor3D{{2.0, -1.0, 0.5}, {4.0, -2.0, 1.0}, {6.0, -3.0, 1.5}} *
        kcal_per_mol
    );
    manostat::Manostat pressure(_settings->manostatSettings);
    pressure.calculatePressure(_box, _data);
    EXPECT_NEAR(
        _data.get_pressure().numerical_value_in(bar),
        0.5 * PRESSURE_FACTOR /
            _box.getVolume().numerical_value_in(Volume::unit),
        1e-12
    );
    // r=(1,2,3), F=(2,-1,0.5): changing the first cell length
    // moves r_x by (1 - 2/sqrt(3)) times the fractional length change.
    EXPECT_NEAR(
        _data.get_coupledPressure().numerical_value_in(bar),
        (2.0 - 4.0 / std::sqrt(3.0)) * PRESSURE_FACTOR /
            _box.getVolume().numerical_value_in(Volume::unit),
        1e-12
    );
}

TEST_F(ManostatRegression, berendsenModesHaveSameHydrostaticVolumeResponse)
{
    constexpr auto increment = 1e-6;
    for (const auto fixed : {FixedAxis::NONE, FixedAxis::Z})
    {
        _settings->manostatSettings.setFixedAxisByUser(fixed);
        _settings->manostatSettings.setCompressibilityByUser(
            increment * per_bar
        );

        // Each mode needs its own settings object: the manostats below hold
        // a reference, not a copy, so sharing one object would make every
        // mode see only the isotropy set last.
        auto isoSettings = _settings->manostatSettings;
        isoSettings.setIsotropyByUser(Isotropy::ISOTROPIC);
        manostat::BerendsenManostat iso(isoSettings);

        auto semiSettings = _settings->manostatSettings;
        semiSettings.setIsotropyByUser(Isotropy::SEMI_ISOTROPIC_XY);
        manostat::SemiIsotropicBerendsenManostat semi(semiSettings);

        auto anisoSettings = _settings->manostatSettings;
        anisoSettings.setIsotropyByUser(Isotropy::ANISOTROPIC);
        manostat::AnisotropicBerendsenManostat aniso(anisoSettings);

        auto fullSettings = _settings->manostatSettings;
        fullSettings.setIsotropyByUser(Isotropy::FULL_ANISOTROPIC);
        manostat::FullAnisotropicBerendsenManostat full(fullSettings);

        for (auto* coupling : std::vector<manostat::BerendsenManostat*>{
                 &iso,
                 &semi,
                 &aniso,
                 &full
             })
        {
            coupling->calculatePressure(_box, _data);
            const auto timeStep = 1.0 * fs;
            const auto mu       = coupling->calculateMu(timeStep);
            EXPECT_NEAR(
                (1.0 - det(mu)).numerical_value_in(one) / increment,
                1.0,
                1e-6
            );
            if (fixed == FixedAxis::Z)
            {
                EXPECT_DOUBLE_EQ(mu[2][2].numerical_value_in(one), 1.0);
            }
        }
    }
}

TEST_F(ManostatRegression, molecularCouplingAllowsAtomicVirial)
{
    _addMolecule({-0.2, 0.2}, 0.0);
    GeneralSettings::setVirialType(VirialType::ATOMIC);
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
            EXPECT_NO_THROW(coupling->applyManostat(
                _box,
                _data,
                toTemperature(0.0),
                1.0 * fs,
                0.1 * angstrom
            ));
            EXPECT_LT(_box.getVolume(), oldVolume);
            EXPECT_NEAR(
                _box.getAtom(1).getPosition()[0].numerical_value_in(angstrom) -
                    _box.getAtom(0).getPosition()[0].numerical_value_in(
                        angstrom
                    ),
                0.4,
                1e-12
            );
        }
    }
    EXPECT_TRUE(testing::internal::GetCapturedStdout().empty());
    manostat::Manostat reportPressure(_settings->manostatSettings);
    EXPECT_NO_THROW(reportPressure.applyManostat(
        _box,
        _data,
        toTemperature(0.0),
        1.0 * fs,
        0.1 * angstrom
    ));
}

TEST_F(ManostatRegression, invalidScalingPreservesSimulationState)
{
    // Each proposal needs its own settings object: the manostats below hold
    // a reference, not a copy, so sharing one object would make every
    // proposal see only the values set last (e.g. overflow/underflow would
    // both end up with cutoff's target pressure and compressibility).
    _settings->manostatSettings.setFixedAxisByUser(FixedAxis::NONE);
    _settings->manostatSettings.setTargetPressureByUser(1.0 * bar);
    _settings->manostatSettings.setTauByUser(1.0 * fs);

    auto negativeRootSettings = _settings->manostatSettings;
    negativeRootSettings.setIsotropyByUser(Isotropy::SEMI_ISOTROPIC_XY);
    negativeRootSettings.setCompressibilityByUser(2.0 * per_bar);
    manostat::SemiIsotropicBerendsenManostat negativeRoot(negativeRootSettings);

    auto negativeVolumeSettings = _settings->manostatSettings;
    negativeVolumeSettings.setIsotropyByUser(Isotropy::ISOTROPIC);
    negativeVolumeSettings.setCompressibilityByUser(2.0 * per_bar);
    manostat::BerendsenManostat negativeVolume(negativeVolumeSettings);

    auto overflowSettings = negativeVolumeSettings;
    overflowSettings.setTargetPressureByUser(-3000.0 * bar);
    overflowSettings.setCompressibilityByUser(1.0 * per_bar);
    manostat::StochasticRescalingManostat overflow(overflowSettings);

    auto underflowSettings = negativeVolumeSettings;
    underflowSettings.setTargetPressureByUser(3000.0 * bar);
    underflowSettings.setCompressibilityByUser(1.0 * per_bar);
    manostat::StochasticRescalingManostat underflow(underflowSettings);

    auto cutoffSettings = negativeVolumeSettings;
    cutoffSettings.setCompressibilityByUser(0.875 * per_bar);
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
        _box.setBox(
            molsys::OrthorhombicBox(
                {10.0 * angstrom, 10.0 * angstrom, 10.0 * angstrom}
            ),
            false
        );
        _addMolecule({4.95, -4.85}, 1.0);
        const auto oldCenter  = _box.getMolecule(0).getCenterOfMass();
        const auto oldDensity = _box.getDensity();
        proposalSettings.at(i)->setIsotropyByUser(
            i == 0 ? Isotropy::SEMI_ISOTROPIC_XY : Isotropy::ISOTROPIC
        );
        EXPECT_THROW_MSG(
            proposals.at(i)->applyManostat(
                _box,
                _data,
                toTemperature(0.0),
                1.0 * fs,
                (i == 4 ? 3.0 : 0.1) * angstrom
            ),
            exc::ManostatException,
            i == 4
                ? "Coulomb radius cut off is larger than half of the minimal "
                  "box dimension"
                : "Invalid manostat scaling matrix"
        );
        EXPECT_EQ(_box.getBoxDimensions(), Pos3D(10.0 * angstrom));
        EXPECT_DOUBLE_EQ(
            _box.getVolume().numerical_value_in(angstrom3),
            1000.0
        );
        EXPECT_DOUBLE_EQ(
            _box.getDensity().numerical_value_in(kg_per_litre),
            oldDensity.numerical_value_in(kg_per_litre)
        );
        EXPECT_DOUBLE_EQ(
            _data.get_volume().numerical_value_in(angstrom3),
            1000.0
        );
        EXPECT_DOUBLE_EQ(
            _data.get_density().numerical_value_in(kg_per_litre),
            oldDensity.numerical_value_in(kg_per_litre)
        );
        EXPECT_EQ(_box.getMolecule(0).getCenterOfMass(), oldCenter);
        EXPECT_DOUBLE_EQ(
            _box.getAtom(0).getPosition()[0].numerical_value_in(angstrom),
            4.95
        );
        EXPECT_DOUBLE_EQ(
            _box.getAtom(1).getPosition()[0].numerical_value_in(angstrom),
            -4.85
        );
        EXPECT_EQ(
            _box.getAtom(0).getVelocity(),
            linalg::Vec3D(1.0, 0.0, 0.0) * angstrom / s
        );
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
    const auto     temperature{toTemperature(300.0)};
    constexpr auto tau             = 1e8;
    constexpr auto compressibility = 4.591e-5;

    ManostatSettings semiSettings;
    semiSettings.setTargetPressureByUser(0.0 * bar);
    semiSettings.setTauByUser(tau * fs);
    semiSettings.setCompressibilityByUser(compressibility * per_bar);
    semiSettings.setIsotropyByUser(Isotropy::SEMI_ISOTROPIC_XY);
    semiSettings.setFixedAxisByUser(FixedAxis::NONE);
    manostat::SemiIsotropicStochasticRescalingManostat semi(semiSettings);

    ManostatSettings freeSettings;
    freeSettings.setTargetPressureByUser(0.0 * bar);
    freeSettings.setTauByUser(tau * fs);
    freeSettings.setCompressibilityByUser(compressibility * per_bar);
    freeSettings.setFixedAxisByUser(FixedAxis::NONE);
    manostat::AnisotropicStochasticRescalingManostat     aniso(freeSettings);
    manostat::FullAnisotropicStochasticRescalingManostat full(freeSettings);

    ManostatSettings fixedSettings;
    fixedSettings.setTargetPressureByUser(0.0 * bar);
    fixedSettings.setTauByUser(tau * fs);
    fixedSettings.setCompressibilityByUser(compressibility * per_bar);
    fixedSettings.setFixedAxisByUser(FixedAxis::Y);
    manostat::FullAnisotropicStochasticRescalingManostat fixed(fixedSettings);

    for (auto* coupling :
         std::vector<manostat::Manostat*>{&semi, &aniso, &full, &fixed})
        coupling->calculatePressure(_box, _data);

    const auto noiseFactor =
        (temperature.quantity_from_zero() * kb * (compressibility * per_bar) /
         (3.0 * tau * _box.getVolume()))
            .numerical_value_in(one);
    const auto      lengthSigma = mstd::sqrt(2.0 * noiseFactor);
    const auto      shearSigma  = mstd::sqrt(4.0 * noiseFactor);
    const auto      volumeSigma = mstd::sqrt(6.0 * noiseFactor);
    NoiseMoments<2> areaHeight;
    NoiseMoments<3> lengths;
    NoiseMoments<3> shape;
    NoiseMoments<3> fixedShape;
    NoiseMoments<1> semiVolume;
    NoiseMoments<1> anisoVolume;
    NoiseMoments<1> fullVolume;
    for (size_t sample = 0; sample < 8192; ++sample)
    {
        const auto timeStep = 1.0 * fs;
        // numerical_value_in(one) can be removed as soon as we have mstd::log
        const auto semiMu =
            semi.calculateMu(_box.getVolume(), temperature, timeStep)
                .numerical_value_in(one);
        const auto anisoMu =
            aniso.calculateMu(_box.getVolume(), temperature, timeStep)
                .numerical_value_in(one);
        const auto fullMu =
            full.calculateMu(_box.getVolume(), temperature, timeStep)
                .numerical_value_in(one);
        const auto fixedMu =
            fixed.calculateMu(_box.getVolume(), temperature, timeStep)
                .numerical_value_in(one);
        areaHeight.add(
            {mstd::log(semiMu[0][0] * semiMu[1][1]) /
                 (2.0 * mstd::sqrt(noiseFactor)),
             mstd::log(semiMu[2][2]) / lengthSigma}
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
    const auto expectedEnergy = _data.get_kineticEnergy() / (scale * scale);
    manostat::Manostat pressure(_settings->manostatSettings);
    pressure.calculatePressure(_box, _data);

    _settings->manostatSettings.setTargetPressureByUser(
        _data.get_pressure() - (3.0 * std::log(scale) * bar)
    );
    _settings->manostatSettings.setTauByUser(1.0 * fs);
    _settings->manostatSettings.setCompressibilityByUser(1.0 * per_bar);
    _settings->manostatSettings.setFixedAxisByUser(FixedAxis::NONE);
    manostat::StochasticRescalingManostat stochastic(_settings->manostatSettings
    );
    stochastic.applyManostat(
        _box,
        _data,
        toTemperature(0.0),
        1.0 * fs,
        0.1 * angstrom
    );

    EXPECT_NEAR(
        (_data.get_kineticEnergy() / expectedEnergy).numerical_value_in(one),
        1.0,
        1e-12
    );
    EXPECT_NEAR(
        tempRatio(_data.get_temperature(), _box.calculateTemperature())
            .numerical_value_in(one),
        1.0,
        1e-12
    );
    EXPECT_NEAR(
        _data.get_momentum()[0].numerical_value_in(g_per_mol * angstrom / fs) /
            (speed * angstrom / s).numerical_value_in(angstrom / fs),
        2.0 / scale,
        1e-12
    );
    EXPECT_NEAR(
        (_data.getKinEnergyMolTensor()[0][0] / expectedEnergy)
            .numerical_value_in(one),
        1.0,
        1e-12
    );
    EXPECT_NEAR(
        (_data.getKinEnergyAtomTensor()[0][0] / expectedEnergy)
            .numerical_value_in(one),
        1.0,
        1e-12
    );

    ResetKineticsSettings resetSettings;
    resetSettings.setFScaleByUser(100);
    resetSettings.setFResetByUser(1);
    resetSettings.setFResetAngularByUser(100);
    kinetics::ResetKinetics reset(resetSettings);
    reset.reset(1, _data, _box, toTemperature(0.0));
    EXPECT_NEAR(
        _box.calculateMomentum()[0].numerical_value_in(
            g_per_mol * angstrom / fs
        ) / (speed * angstrom / s).numerical_value_in(angstrom / fs),
        0.0,
        1e-12
    );
}

TEST_F(ManostatRegression, atomicVirialSupportsMonatomicPressureCoupling)
{
    _addMolecule({-1.0}, 1e13);
    _addMolecule({1.0}, -1e13);
    _data.calculateKinetics(_box);
    GeneralSettings::setVirialType(VirialType::ATOMIC);
    _settings->manostatSettings.setCompressibilityByUser(0.001 * per_bar);
    manostat::BerendsenManostat berendsen(_settings->manostatSettings);
    manostat::StochasticRescalingManostat stochastic(_settings->manostatSettings
    );
    EXPECT_NO_THROW(berendsen.applyManostat(
        _box,
        _data,
        toTemperature(300.0),
        0.5 * fs,
        0.1 * angstrom
    ));
    EXPECT_NO_THROW(stochastic.applyManostat(
        _box,
        _data,
        toTemperature(300.0),
        0.5 * fs,
        0.1 * angstrom
    ));
    EXPECT_TRUE(isfinite(_box.getVolume()));
    EXPECT_GT(_box.getVolume().numerical_value_in(Volume::unit), 0.0);
}
