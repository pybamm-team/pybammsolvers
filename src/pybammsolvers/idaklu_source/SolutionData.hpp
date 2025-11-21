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
    SolutionData() : ownership_transferred(false) {}

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
      yterm_return(yterm_return),
      ownership_transferred(false)
    {}

    /**
     * @brief Destructor - cleans up if generate_solution() was never called
     */
    ~SolutionData() {
      if (!ownership_transferred) {
        delete[] t_return;
        delete[] y_return;
        delete[] yp_return;
        delete[] yS_return;
        delete[] ypS_return;
        delete[] yterm_return;
      }
    }


    /**
     * @brief Deleted copy constructor (prevent double-free of raw pointers)
     */
    SolutionData(const SolutionData &solution_data) = delete;

    /**
     * @brief Deleted copy assignment (prevent double-free of raw pointers)
     */
    SolutionData& operator=(const SolutionData &solution_data) = delete;

    /**
     * @brief Move constructor (transfer ownership of pointers)
     */
    SolutionData(SolutionData &&solution_data) noexcept 
      : flag(solution_data.flag),
        number_of_timesteps(solution_data.number_of_timesteps),
        length_of_return_vector(solution_data.length_of_return_vector),
        arg_sens0(solution_data.arg_sens0),
        arg_sens1(solution_data.arg_sens1),
        arg_sens2(solution_data.arg_sens2),
        length_of_final_sv_slice(solution_data.length_of_final_sv_slice),
        save_hermite(solution_data.save_hermite),
        t_return(solution_data.t_return),
        y_return(solution_data.y_return),
        yp_return(solution_data.yp_return),
        yS_return(solution_data.yS_return),
        ypS_return(solution_data.ypS_return),
        yterm_return(solution_data.yterm_return),
        ownership_transferred(solution_data.ownership_transferred) {
      // Nullify source pointers to prevent double-free
      solution_data.t_return = nullptr;
      solution_data.y_return = nullptr;
      solution_data.yp_return = nullptr;
      solution_data.yS_return = nullptr;
      solution_data.ypS_return = nullptr;
      solution_data.yterm_return = nullptr;
      solution_data.ownership_transferred = true;
    }

    /**
     * @brief Move assignment (transfer ownership of pointers)
     */
    SolutionData& operator=(SolutionData &&solution_data) noexcept {
      if (this != &solution_data) {
        // Clean up existing data
        if (!ownership_transferred) {
          delete[] t_return;
          delete[] y_return;
          delete[] yp_return;
          delete[] yS_return;
          delete[] ypS_return;
          delete[] yterm_return;
        }
        
        // Transfer ownership
        flag = solution_data.flag;
        number_of_timesteps = solution_data.number_of_timesteps;
        length_of_return_vector = solution_data.length_of_return_vector;
        arg_sens0 = solution_data.arg_sens0;
        arg_sens1 = solution_data.arg_sens1;
        arg_sens2 = solution_data.arg_sens2;
        length_of_final_sv_slice = solution_data.length_of_final_sv_slice;
        save_hermite = solution_data.save_hermite;
        t_return = solution_data.t_return;
        y_return = solution_data.y_return;
        yp_return = solution_data.yp_return;
        yS_return = solution_data.yS_return;
        ypS_return = solution_data.ypS_return;
        yterm_return = solution_data.yterm_return;
        ownership_transferred = solution_data.ownership_transferred;
        
        // Nullify source pointers to prevent double-free
        solution_data.t_return = nullptr;
        solution_data.y_return = nullptr;
        solution_data.yp_return = nullptr;
        solution_data.yS_return = nullptr;
        solution_data.ypS_return = nullptr;
        solution_data.yterm_return = nullptr;
        solution_data.ownership_transferred = true;
      }
      return *this;
    }

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
    bool ownership_transferred;  // Track if pointers have been transferred to Python
};

#endif // PYBAMM_IDAKLU_SOLUTION_DATA_HPP
