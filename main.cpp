#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include <chrono>
#include <string>
#include <nlopt.hpp>
#include "utils.h"
using namespace std;
/*
h-bar = 1
m = 1
omega = 1
*/


int main(int argc, char *argv[]) {

    ///////// Variable declarations and default values //////////////////
    std::vector<double> wfParams = std::vector<double>{0.5, 1.};
    //set a maximum number of iterations for gd
    unsigned int  nMaxIter = 3E0;
    //set tolerance for convergence of gd
    double  energyTol = 1E-7;
    //parameters of the simulation
    SimulationParams simPar ;
    //hyperparameters for gradient descent:
    GdParams gd_parameters ;
    if (argc > 1){
        // Input filename from command line
        std::string input_filename = argv[1];
        std::string path_input = "./Input/";

        // create filestream to read in inputs from file
        std::ifstream infile;
        infile.open(path_input + input_filename);

        if (!infile.is_open())
        {
            std::cout << "Error opening file" << std::endl;
            return 1;
        }

        // read in inputs from file

        std::string line;

        while (std::getline(infile, line))
        {
            // loop through lines and sett corect variables. name and value separated by "="
            std::string name = line.substr(0, line.find("="));
            std::string value = line.substr(line.find("=") + 1);
            if (value == "")
            {}
            else if (name == "learning_rate")
            {gd_parameters.learning_rate = std::stod(value);}
            else if (name == "momentum")
            {gd_parameters.momentum = std::stod(value);}
            else if (name == "alpha")
            {wfParams[0] = std::stod(value);}
            else if (name == "beta")
            {wfParams[1] = std::stod(value);}
            else if (name == "gamma")
            {simPar.gamma = std::stod(value);}
            else if (name == "nMaxIter")
            {nMaxIter = std::stoi(value);}
            else if (name == "energyTol")
            {energyTol = std::stod(value);}
            else if (name == "calculateGradients")
            {simPar.calculateGradients = (bool)std::stoi(value);}
            else if (name == "numberOfParticles")
            {simPar.numberOfParticles = std::stoi(value);}
            else if (name == "numberOfMetropolisSteps")
            {simPar.numberOfMetropolisSteps = std::stoi(value);}
            else if (name == "numberOfEquilibrationSteps")
            {simPar.numberOfEquilibrationSteps = std::stoi(value);}
            else if (name == "omega")
            {simPar.omega = std::stod(value);}
            else if (name == "stepLength")
            {simPar.stepLength = std::stod(value);}
            else if (name == "filename")
            {simPar.filename = "Outputs/" + value;}
            else if (name == "numberOfDimensions")
            {simPar.numberOfDimensions=std::stoi(value);}
            else
            {
                std::cout << "Error reading file." << std::endl;
                return 1;
            }
        }
    }
    else{
        std::cout << "WARNING: No settings file provided" << std::endl;
        //DEFINE SIMULATION PARAMETERS
        //no trainable params
        simPar.numberOfDimensions=3;
        simPar.numberOfParticles=10;//50; 100
        simPar.numberOfMetropolisSteps=5E4;
        simPar.numberOfEquilibrationSteps=1E3;
        simPar.calculateGradients=true;
        simPar.omega=1;
        simPar.gamma=1;
        simPar.stepLength=5E-1;
        simPar.filename="Outputs/output.txt";
        gd_parameters.learning_rate = 3E-3;
        gd_parameters.momentum = 0.6;
    }
    simPar.a_ho = std::sqrt(1./simPar.omega); // Characteristic size of the Harmonic Oscillator
    simPar.stepLength *= simPar.a_ho; // Scale the steplength in case of changed omega        

        std::cout << "Simulation Parameters:" << std::endl;
    std::cout << "Number of Dimensions: " << simPar.numberOfDimensions << std::endl;
    std::cout << "Number of Particles: " << simPar.numberOfParticles << std::endl;
    std::cout << "Number of Metropolis Steps: " << simPar.numberOfMetropolisSteps << std::endl;
    std::cout << "Number of Equilibration Steps: " << simPar.numberOfEquilibrationSteps << std::endl;
    std::cout << "Calculate Gradients: " << simPar.calculateGradients << std::endl;
    std::cout << "Omega: " << simPar.omega << std::endl;
    std::cout << "Gamma: " << simPar.gamma << std::endl;
    std::cout << "Step Length: " << simPar.stepLength << std::endl;
    std::cout << "Filename: " << simPar.filename << std::endl;

    std::cout << std::endl;

    std::cout << "Gradient Descent Parameters:" << std::endl;
    std::cout << "Learning Rate: " << gd_parameters.learning_rate << std::endl;
    std::cout << "Momentum: " << gd_parameters.momentum << std::endl;
    std::cout << "calculate grads? " << simPar.calculateGradients << std::endl;

    //#define TIMEING // Comment out turn off timing
    #ifdef TIMEING
    auto times = vector<int>();
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;
    auto t1 = high_resolution_clock::now();
    #endif

    double optimal_energy; 
    // LBFGS ALGORITHM
    nlopt::opt opt(nlopt::LD_LBFGS, 2);
    // GRADIENT DESCENT WITH MOMENTUM
    // momentumOptimizer opt(2, &gd_parameters);

    opt.set_min_objective(wrapSimulation, (void *) & simPar );
    opt.set_maxeval(nMaxIter);
    opt.set_ftol_abs(energyTol);
    auto optimal_params = opt.optimize(wfParams, optimal_energy);

    #ifdef TIMEING
    auto t2 = high_resolution_clock::now();
    /* Getting number of milliseconds as an integer. */
    auto ms_int = duration_cast<milliseconds>(t2 - t1);
    times.push_back(ms_int.count());
    #endif
    #ifdef TIMEING
    cout << "times : " << endl;
    for(unsigned int i = 0; i<times.size(); i++)
        cout << times[i] << endl;
    #endif

    return 0;
}
