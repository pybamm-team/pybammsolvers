#ifndef PYBAMM_IDAKLU_REDUCE_HPP
#define PYBAMM_IDAKLU_REDUCE_HPP

#include "common.hpp"
#include "KnotReducer.hpp"
#include "SolutionData.hpp"  // for vector_to_numpy()
#include <cstring>

/**
 * @brief Post-hoc streaming knot reduction on multi-segment solution data.
 *
 * Accepts vectors of per-segment arrays in the flat time-major layout
 * used by pybamm.Solution from IDAKLUSolver:
 *   ts[i]:  1D, shape (M_i,)
 *   ys[i]:  1D, shape (M_i * N_i,), time-major
 *   yps[i]: 1D, shape (M_i * N_i,), same layout
 *   atols[i]: 1D, shape (N_i,), per-state absolute tolerance
 *
 * n_states per segment is inferred as ys[i].size() / ts[i].size().
 *
 * @return py::tuple of three VectorRealtypeNdArray (reduced ts, ys, yps).
 */
inline py::object reduce_knots(
    const std::vector<np_array_realtype>& ts,
    const std::vector<np_array_realtype>& ys,
    const std::vector<np_array_realtype>& yps,
    const std::vector<np_array_realtype>& atols,
    double rtol,
    double multiplier)
{
    const size_t n_seg = ts.size();

    auto* out_ts  = new std::vector<np_array_realtype>();
    auto* out_ys  = new std::vector<np_array_realtype>();
    auto* out_yps = new std::vector<np_array_realtype>();
    out_ts->reserve(n_seg);
    out_ys->reserve(n_seg);
    out_yps->reserve(n_seg);

    for (size_t seg = 0; seg < n_seg; ++seg) {
        const sunrealtype* t_ptr = ts[seg].data();
        const sunrealtype* y_ptr = ys[seg].data();
        const sunrealtype* yp_ptr = yps[seg].data();
        const sunrealtype* atol_ptr = atols[seg].data();

        const int M = static_cast<int>(ts[seg].size());
        const int N = M > 0 ? static_cast<int>(ys[seg].size()) / M : 0;

        std::vector<sunrealtype> out_t;
        std::vector<sunrealtype> out_y;
        std::vector<sunrealtype> out_yp;
        out_t.reserve(M);
        out_y.reserve(static_cast<size_t>(M) * N);
        out_yp.reserve(static_cast<size_t>(M) * N);

        StreamingKnotReducer reducer(N, rtol, atol_ptr,
                                     multiplier,
                                     out_t, out_y, out_yp);
        for (int i = 0; i < M; ++i) {
            reducer.ProcessPoint(
                t_ptr[i],
                &y_ptr[static_cast<size_t>(i) * N],
                &yp_ptr[static_cast<size_t>(i) * N],
                false
            );
        }
        reducer.Finalize();

        out_ts->push_back(vector_to_numpy(std::move(out_t)));
        out_ys->push_back(vector_to_numpy(std::move(out_y)));
        out_yps->push_back(vector_to_numpy(std::move(out_yp)));
    }

    // Wrap in capsules so Python owns the lifetime
    py::capsule cap_ts(out_ts, [](void* v) {
        delete static_cast<std::vector<np_array_realtype>*>(v);
    });
    py::capsule cap_ys(out_ys, [](void* v) {
        delete static_cast<std::vector<np_array_realtype>*>(v);
    });
    py::capsule cap_yps(out_yps, [](void* v) {
        delete static_cast<std::vector<np_array_realtype>*>(v);
    });

    // Return as Python-visible VectorRealtypeNdArray (opaque bind_vector)
    return py::make_tuple(
        py::cast(std::move(*out_ts)),
        py::cast(std::move(*out_ys)),
        py::cast(std::move(*out_yps))
    );
}

#endif // PYBAMM_IDAKLU_REDUCE_HPP
