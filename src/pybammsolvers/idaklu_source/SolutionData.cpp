#include "SolutionData.hpp"

Solution SolutionData::generate_solution() {
  py::capsule free_t_when_done(
    t_return,
    [](void *f) {
      realtype *vect = reinterpret_cast<realtype *>(f);
      delete[] vect;
    }
  );

  np_array t_ret = np_array(
    number_of_timesteps,
    &t_return[0],
    free_t_when_done
  );

  py::capsule free_y_when_done(
    y_return,
    [](void *f) {
      realtype *vect = reinterpret_cast<realtype *>(f);
      delete[] vect;
    }
  );

  np_array y_ret = np_array(
    number_of_timesteps * length_of_return_vector,
    &y_return[0],
    free_y_when_done
  );

  py::capsule free_yS_when_done(
    yS_return,
    [](void *f) {
      realtype *vect = reinterpret_cast<realtype *>(f);
      delete[] vect;
    }
  );

  np_array yS_ret = np_array(
    std::vector<ptrdiff_t> {
      arg_sens0,
      arg_sens1,
      arg_sens2
    },
    &yS_return[0],
    free_yS_when_done
  );

  // Final state slice, yterm
  py::capsule free_yterm_when_done(
    yterm_return,
    [](void *f) {
      realtype *vect = reinterpret_cast<realtype *>(f);
      delete[] vect;
    }
  );

  np_array y_term = np_array(
    length_of_final_sv_slice,
    &yterm_return[0],
    free_yterm_when_done
  );

  // Store the solution
  return Solution(flag, t_ret, y_ret, yS_ret, y_term);
}
