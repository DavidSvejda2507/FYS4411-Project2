#pragma once
#include<vector>
#include <memory>
#include"Samplers/sampler.h"
//define a struct as a container of simulation params.
struct simulationParams{
    unsigned int numberOfDimensions;
    unsigned int numberOfParticles;
    unsigned int numberOfMetropolisSteps;
    unsigned int numberOfEquilibrationSteps;
    bool calculateGradients;
    double omega;
    double gamma;
    double a_ho;
    double stepLength;
    std::string filename;
    int base_seed;//store base seed , will be manipulated in different ways in every simulation
};
struct gdParams{
    double learning_rate;
    double momentum;
};

//create some aliases
typedef struct simulationParams SimulationParams;
typedef struct gdParams GdParams;
//define functions which run a single MCMC simulation and wrap it so it can be used by an optimizer.
double wrapSimulation(  
    const std::vector<double> &params,
    std::vector<double> &grad, 
    void * xPtr  );
std::unique_ptr<Sampler> runSimulation(
    SimulationParams * P,
    std::vector<double> params
);
//define function which runs MC simulation without gradients.
void wrapSimulationLargeScale(const std::vector<double> &params,  void * xPtr);

//define an alias for a pointer to an objective function to be minimized.
typedef double (*obj_func)(const std::vector<double>& trainable_params,std::vector<double> &grad, void * xPtr);

//define a class for momentum gd which mimics nlopt library's class nlopt::opt
class momentumOptimizer {
  public:
    momentumOptimizer(int n_params, GdParams* gd_params );
    double optimize(std::vector<double>& x, double& opt_f )  ;  //will call func recursively and do gd
    void set_min_objective( obj_func, void * params );          //sets the function to minimize and the parameters to use it with.
    void set_maxeval(unsigned int maxeval){m_maxeval=maxeval;} //set stopping condition
    void set_ftol_abs(double tol){m_f_abs_tol=tol;}             //set stopping condition
  protected:
    void * m_function_params;   //will store a ptr to a struct simulationParams containing the simulation parameters.
    obj_func m_objective_function; //function to minimize (see typedef above)

    //declare variables for gradient descent
    double m_learning_rate;     
    double m_momentum;
    std::vector<double> m_velocity;
    std::vector<double> m_gradient;
    bool m_initiated;//flag to guarantee obj function initiated properly
    //stopping params of gd
    unsigned int m_maxeval=1000;
    double m_f_abs_tol=0.01; 
};
unsigned int *** init_3d_array(unsigned int nx,unsigned int ny,unsigned int nz);