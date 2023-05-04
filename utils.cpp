#include "utils.h"
#include "Hamiltonians/harmonicoscillator.h"
#include "Hamiltonians/harmonicoscillator3d.h"
#include "Hamiltonians/harmonicoscillatorcoulomb.h"
#include "InitialStates/initialstate.h"
#include "Math/random.h"
#include "Samplers/sampler.h"
#include "Solvers/metropolis.h"
#include "Solvers/metropolisHastings.h"
#include "WaveFunctions/interactinggaussian.h"
#include "WaveFunctions/interactinggaussian2fermion.h"
#include "WaveFunctions/interactinggaussian3d.h"
#include "WaveFunctions/interactinggaussianfermion.h"
#include "WaveFunctions/simplegaussian.h"
#include "WaveFunctions/simplegaussian3d.h"
#include "WaveFunctions/testwavefunction.h"
#include "omp.h"
#include "particle.h"
#include "system.h"
#include <memory>
#include <nlopt.hpp>
#include <vector>

std::seed_seq seq{std::random_device{}()};

double wrapSimulation(const std::vector<double> &params, std::vector<double> &grad, void *xPtr)
{
    // wraps the runSimulation function so that it can be used with nlopt library
    // check if output file already exists, if so,
    // set file_initiated to true (avoid printing header line over and over in outputs.txt)
    // just to handle the .txt easier to handle.

    // convert P to struct pointer:
    SimulationParams *P = (SimulationParams *)xPtr;
    // need a new seed every time a simulation is run.
    std::vector<int> current_seed(1);
    seq.generate(current_seed.begin(), current_seed.end());
    P->base_seed = current_seed[0];
    bool file_initiated;
    if (FILE *fptr = fopen(P->filename.c_str(), "r"))
    {
        fclose(fptr);
        file_initiated = true;
    }
    else
        file_initiated = false;
    int NUM_THREADS = omp_get_max_threads();
    NUM_THREADS = 1;
    std::cout << "Using " << NUM_THREADS << " threads." << std::endl;
    std::vector<std::unique_ptr<class Sampler>> samplers(NUM_THREADS);

    ///////////////////////////////////////////////
    ///// START PARALLEL REGION //////////////////
    ///////////////////////////////////////////////

#pragma omp parallel num_threads(NUM_THREADS)
    {
        int thread_number = omp_get_thread_num();

        auto sampler = runSimulation(
            P,
            params // params to init wavefunc
        );
        samplers[thread_number] = std::move(sampler);
    }
    ///////////////////////////////////////////////
    ///// END PARALLEL REGION //////////////////
    ///////////////////////////////////////////////
    // gather all simulation results in one sampler
    std::unique_ptr<class Sampler> collective_sampler = std::make_unique<class Sampler>(samplers);
    // write to file
    if (!file_initiated)
    {
        collective_sampler->initiateFile(P->filename);
        file_initiated = true;
    }
    collective_sampler->writeToFile(P->filename);
    // write to terminal:
    collective_sampler->printOutputToTerminalShort();

    // compute energy difference

    double energy = collective_sampler->getEnergy();

    // sampler computes gradient
    grad = collective_sampler->computeGradientEtrial();

    return energy;
}

// to keep main clean, define the large scale simulation here.
void wrapSimulationLargeScale(const std::vector<double> &params, void *xPtr)
{
    // xPtr points to a struct SimulationParams.
    // wraps the runSimulation function so that it is used to run a large monte carlo simulation,
    // no gradient is computed now.
    // NB need to set compute gradients to 0 in the input file.

    // convert P to struct pointer:
    SimulationParams *P = (SimulationParams *)xPtr;
    int NUM_THREADS = omp_get_max_threads();
    std::cout << "Using " << NUM_THREADS << " threads." << std::endl;
    std::vector<std::unique_ptr<class Sampler>> samplers(NUM_THREADS);

    ///////////////////////////////////////////////
    ///// START PARALLEL REGION //////////////////
    ///////////////////////////////////////////////

#pragma omp parallel
    {
        int thread_number = omp_get_thread_num();

        auto sampler = runSimulation(
            P,
            params // params to init wavefunc
        );

        sampler->writeHistogram();
        samplers[thread_number] = std::move(sampler);
    }
    ///////////////////////////////////////////////
    ///// END PARALLEL REGION //////////////////
    ///////////////////////////////////////////////

    return;
}

std::unique_ptr<class Sampler> runSimulation(
    SimulationParams *P,
    std::vector<double> params)
{

    int seed = (P->base_seed) + 77 * omp_get_thread_num();
    // The random engine can also be built without a seed
    auto rng = std::make_unique<Random>(seed);
    // Initialize particles
    // auto particles = setupNonOverlappingGaussianInitialState(numberOfDimensions, numberOfParticles, *rng, a_ho);

    auto particles = setupNonOverlappingGaussianInitialState(P->numberOfDimensions, P->numberOfParticles, *rng, P->a_ho);
    // Construct a unique pointer to a new System

    auto system = std::make_unique<System>(
        // Construct unique_ptr to Hamiltonian
        // std::make_unique<HarmonicOscillator>(omega),
        std::make_unique<HormonicOscillatorCoulomb>(P->omega),
        // Construct unique_ptr to wave function
        // std::make_unique<TestWavefunction>(
        // std::make_unique<SimpleGaussian>(params[0])),
        // std::make_unique<SimpleGaussian3D>(params[0], params[1])),
        // std::make_unique<InteractingGaussian>(params[0])),
        // std::make_unique<InteractingGaussian3D>(params[0], params[1])),
        // std::make_unique<InteractingGaussian2Fermion>(params[0], params[1])),
        // std::make_unique<InteractingGaussianFermion>(params[0], params[1])),

        // std::make_unique<SimpleGaussian>(params[0]),
        // std::make_unique<SimpleGaussian3D>(params[0], params[1]),
        // std::make_unique<InteractingGaussian>(params[0]),
        // std::make_unique<InteractingGaussian3D>(params[0], params[1]),
        // std::make_unique<InteractingGaussian2Fermion>(params[0], params[1]),
        std::make_unique<InteractingGaussianFermion>(params[0], params[1]),
        // Construct unique_ptr to solver, and move rng
        std::make_unique<MetropolisHastings>(std::move(rng)),
        // std::make_unique<Metropolis>(std::move(rng)),
        //  Move the vector of particles to system
        std::move(particles),
        P->calculateGradients);
    // Run steps to equilibrate particles
    std::unique_ptr<class Sampler> sampler = system->runEquilibrationSteps(
        P->stepLength,
        P->numberOfEquilibrationSteps);

    // Run the Metropolis algorithm
    sampler = system->runMetropolisSteps(
        std::move(sampler),
        P->stepLength,
        P->numberOfMetropolisSteps);

    return sampler;
}

// DEFINITION OF GRADIENT DESCENT WITH MOMENTUM, HERE ARE ALL MEMBER DEFINITION FOR THE CLASS momentumOpimizer.
momentumOptimizer::momentumOptimizer(int n_params, GdParams *gd_params)
{
    m_learning_rate = gd_params->learning_rate;
    m_momentum = gd_params->momentum;
    m_velocity = std::vector<double>(n_params, 0.0);
    m_gradient = std::vector<double>(n_params, 0.0);
    m_initiated = false;
}
void momentumOptimizer::set_min_objective(obj_func funct, void *func_data)
{
    m_objective_function = funct;
    m_function_params = func_data;
    m_initiated = true;
}
double momentumOptimizer::optimize(std::vector<double> &x, double &opt_f)
{
    if (!m_initiated)
    {
        std::cout << "Error. Objective function not set. Terminating program." << std::endl;
        exit(1);
    }
    unsigned int iterCount = 0;
    double energyChange = 1; // set to 1 just to enter while loop, should be >= energyTol
    double oldEnergy = 1E7;  // to enter while loop twice
    while ((iterCount < m_maxeval) && (energyChange >= m_f_abs_tol))
    {

        opt_f = wrapSimulation(x, m_gradient, m_function_params);

        energyChange = fabs(oldEnergy - opt_f);

        oldEnergy = opt_f;

        // update parameters using momentum gd
        for (unsigned int i = 0; i < m_gradient.size(); i++)
        {

            m_velocity[i] = m_momentum * m_velocity[i] - m_learning_rate * m_gradient[i];
            x[i] += m_velocity[i];
        }

        iterCount++;
    }

    return opt_f;
}