#ifndef PYBAMM_IDAKLU_SOLUTION_DATA_HPP
#define PYBAMM_IDAKLU_SOLUTION_DATA_HPP


#include "common.hpp"
#include "Solution.hpp"

/**
 * @brief SolutionData class. Contains all the data needed to create a Solution
 */
class SolutionData
{
  public:
    /**
     * @brief Default constructor
     */
    SolutionData() = default;

    /**
     * @brief constructor using fields
     */
    SolutionData(
      const int flag,
      const int number_of_timesteps,
      const int length_of_return_vector,
      const int arg_sens0,
      const int arg_sens1,
      const int arg_sens2,
      const int length_of_final_sv_slice,
      const bool save_hermite,
      sunrealtype *t_return,
      sunrealtype *y_return,
      sunrealtype *yp_return,
      sunrealtype *yS_return,
      sunrealtype *ypS_return,
      sunrealtype *yterm_return):
      flag(flag),
      number_of_timesteps(number_of_timesteps),
      length_of_return_vector(length_of_return_vector),
      arg_sens0(arg_sens0),
      arg_sens1(arg_sens1),
      arg_sens2(arg_sens2),
      length_of_final_sv_slice(length_of_final_sv_slice),
      save_hermite(save_hermite),
      t_return(t_return),
      y_return(y_return),
      yp_return(yp_return),
      yS_return(yS_return),
      ypS_return(ypS_return),
      yterm_return(yterm_return)
    {}


    /**
     * @brief Default copy from another SolutionData
     */
    SolutionData(const SolutionData &solution_data) = default;

    /**
     * @brief Create a solution object from this data
     */
    Solution generate_solution();

private:

    int flag;
    int number_of_timesteps;
    int length_of_return_vector;
    int arg_sens0;
    int arg_sens1;
    int arg_sens2;
    int length_of_final_sv_slice;
    bool save_hermite;
    sunrealtype *t_return;
    sunrealtype *y_return;
    sunrealtype *yp_return;
    sunrealtype *yS_return;
    sunrealtype *ypS_return;
    sunrealtype *yterm_return;
};

#endif // PYBAMM_IDAKLU_SOLUTION_DATA_HPP
