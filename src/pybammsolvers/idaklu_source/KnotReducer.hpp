#ifndef PYBAMM_KNOT_REDUCER_HPP
#define PYBAMM_KNOT_REDUCER_HPP

#include "common.hpp"
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>

/**
 * @brief True streaming knot reducer with Bernstein-certified error bounds
 *        and optional least-squares derivative refinement.
 *
 * Processes points one at a time as IDA generates them, deciding inline
 * whether to keep or discard each point. Only committed points are stored.
 *
 * ERROR CHECKING uses a three-level hierarchy for maximum efficiency:
 *
 *   Level 1 (hot path): Conservative Bernstein bound with inv_atol weighting.
 *     Division-free, fully SIMD-vectorizable. Catches the common case where
 *     the merge succeeds easily. Endpoint sub-intervals use only 2 of the 4
 *     Bernstein control points (anchor/new-point error is exactly zero).
 *
 *   Level 2 (rare): Bernstein bound with exact WRMS weight.
 *     Uses fmin(|y_left|, |y_right|) for conservative weight certification.
 *     Only entered when Level 1's conservative bound exceeds the threshold.
 *
 *   Level 3 (very rare): De Casteljau midpoint subdivision tightening.
 *     Splits each Bernstein curve at the midpoint via de Casteljau's algorithm,
 *     producing 8 refined control points whose convex hull is ~4x tighter.
 *
 * LS DERIVATIVE REFINEMENT (optional, integral L² objective):
 *
 *   After the greedy pass selects knots, a least-squares update adjusts the
 *   derivative (y') values at each knot to minimize the continuous integral
 *   of the squared error between the original and reduced Hermite interpolants:
 *
 *     min_δ  Σ_spans ∫ (H_orig(t) - H_new(t; y'+δ))² dt
 *
 *   The sensitivity functions φ_A(t) = h₁₀(u)·hₘ and φ_B(t) = h₁₁(u)·hₘ
 *   are global cubics on each merged span, so their inner products come from
 *   the Hermite mass matrix and are CLOSED-FORM constants:
 *
 *     diag += hₘ³/105,   offdiag += -hₘ³/140
 *
 *   The matrix is strictly diagonally dominant (1/105 > 1/140), hence always
 *   SPD — no Tikhonov regularization needed, no zero-diagonal branching.
 *
 *   FACTORED ACCUMULATION: By expanding knot errors d_k = y_k - H(t_k) and
 *   absorbing the constant endpoint data into scalar accumulators, the per-
 *   state inner loop reduces from 7 FMA (unfactored pointwise) to 4 FMA:
 *
 *     sum_A[s] += c_d_A · y_p[s] + c_dp_A · y'_p[s]   (2 FMA)
 *     sum_B[s] += c_d_B · y_p[s] + c_dp_B · y'_p[s]   (2 FMA)
 *
 *   The Thomas solve at Finalize() is O(K) scalar factorization + O(K*S)
 *   fused back-sub + yp update — negligible versus O(M*S) accumulation.
 *
 *   GUARANTEE: The refinement can only reduce or maintain the L² error.
 *   δ=0 is always feasible (produces the original y' values), so the LS
 *   minimizer can never increase the error.
 *
 * NUMERICAL STABILITY: The merged Hermite spline is evaluated using normalized
 * coordinates u = (t - t_anchor) / h_merged with Hermite basis functions that
 * are O(1) for u in [0,1]. This avoids the catastrophic cancellation inherent
 * in Horner evaluation of monomial coefficients that scale as O(1/h^k).
 *
 * Memory:
 *   - O(1) for anchor point
 *   - O(span) for intermediate points in current merge window
 *   - O(K) for committed output where K = kept points
 *   - O(K + K*S) for LS refinement arrays (when enabled): K scalars for
 *     state-independent matrix + K*S for per-state RHS
 *
 * Algorithm:
 *   - anchor: last committed point (fixed endpoint of merged spline)
 *   - window: all points since anchor (to be potentially merged)
 *   - When new point arrives, check if window + new can all be replaced by anchor->new
 *   - If yes: add new to window
 *   - If no: commit last point in window, it becomes anchor, continue
 */
class StreamingKnotReducer {
public:
    StreamingKnotReducer(
        int n_states,
        double rtol,
        const double* atol_ptr,
        double multiplier,
        // Output arrays (solver's arrays, will grow as points are committed)
        std::vector<sunrealtype>& out_t,
        std::vector<sunrealtype>& out_y,
        std::vector<sunrealtype>& out_yp
    ) : n_states_(n_states),
        rtol_(rtol),
        threshold_(static_cast<double>(n_states) * (multiplier - 1.0) * (multiplier - 1.0)),
        active_(multiplier > 1.0),
        ls_knot_count_(0),
        ls_interval_start_(0),
        out_t_(&out_t),
        out_y_(&out_y),
        out_yp_(&out_yp),
        out_count_(0),
        has_anchor_(false)
    {
        // Precompute 1/atol (solver's raw tolerance, no multiplier scaling)
        inv_atol_.resize(n_states);
        for (int j = 0; j < n_states; ++j) {
            inv_atol_[j] = 1.0 / atol_ptr[j];
        }

        // Double-buffered knot error caches for Bernstein bound computation
        dk_buf0_.resize(n_states);
        dpk_buf0_.resize(n_states);
        dk_buf1_.resize(n_states);
        dpk_buf1_.resize(n_states);

        // Sentinel states: top-K by inv_atol (most sensitive to error).
        // Used for O(K) early-exit on reject paths before the full O(n) SIMD loop.
        n_sentinels_ = std::min(n_states, 32);
        sentinels_.resize(n_sentinels_);
        {
            std::vector<int> idx(n_states);
            for (int j = 0; j < n_states; ++j) idx[j] = j;
            std::partial_sort(idx.begin(), idx.begin() + n_sentinels_, idx.end(),
                [this](int a, int b) { return inv_atol_[a] > inv_atol_[b]; });
            for (int s = 0; s < n_sentinels_; ++s) sentinels_[s] = idx[s];
        }

        // Reserve space for window (will grow dynamically)
        window_t_.reserve(100);
        window_y_.reserve(100 * n_states);
        window_yp_.reserve(100 * n_states);

        // LS work buffers (factored accumulation)
        ls_sum_A_.resize(n_states, 0.0);
        ls_sum_B_.resize(n_states, 0.0);
    }

    bool IsActive() const { return active_; }

    /**
     * @brief Process a new point from IDA.
     *
     * Matches post-processing greedy algorithm:
     * - Track "candidate" = last point for which merge succeeded
     * - On failure, commit candidate, anchor = candidate, start new window
     *
     * When LS refinement is enabled, the closing span's interior points
     * are accumulated into the tri-diagonal normal equations just before
     * the candidate is committed (cache-hot from the merge check).
     */
    void ProcessPoint(sunrealtype t, const sunrealtype* y, const sunrealtype* yp, bool is_breakpoint) {
        // First point ever: commit as anchor
        if (!has_anchor_) {
            CommitPoint(t, y, yp);
            SetAnchor(t, y, yp);
            return;
        }

        // Breakpoint: must commit candidate (if any) and this point
        if (is_breakpoint) {
            if (!window_t_.empty()) {
                FlushCandidate();
                // Matrix entries for the trailing span: candidate → breakpoint
                LSAccumulateMatrix(t - window_t_.back());
            } else {
                // Matrix entries for the span: anchor → breakpoint (no interior)
                LSAccumulateMatrix(t - anchor_t_);
            }
            CommitPoint(t, y, yp);
            SetAnchor(t, y, yp);
            // Back-to-back anchor: finalize LS for the completed interval
            FinalizeInterval();
            return;
        }

        // First point after anchor: just add to window
        // Try to extend window with new point
        if (window_t_.empty() || CanMergeWindow(t, y, yp)) {
            AddToWindow(t, y, yp);
            return;
        }

        // Failure: commit candidate (= window.back()), it becomes new anchor
        const bool back_to_back = (window_t_.size() == 1);
        FlushCandidate();
        const size_t last = window_t_.size() - 1;
        SetAnchor(window_t_[last], &window_y_[last * n_states_],
                    &window_yp_[last * n_states_]);
        // Back-to-back anchor (window had only 1 point = candidate):
        // finalize LS since there are no interior points to couple across
        if (back_to_back) {
            FinalizeInterval();
        }
        AddToWindow(t, y, yp);
    }

    /**
     * @brief Flush remaining candidate at end of solve, then finalize
     *        the last interval's LS refinement.
     */
    void Finalize() {
        if (!window_t_.empty()) {
            FlushCandidate();
        }
        FinalizeInterval();
    }

    /**
     * @brief Solve LS refinement for the current continuous interval,
     *        then reset the LS system for the next interval.
     *
     * Called at every interval boundary: integrator reinitialize (t_eval),
     * events, and end-of-solve. Ensures the LS tridiagonal system is
     * scoped to a single continuous interval — never spanning across
     * discontinuities where the integrator reinitializes.
     *
     * The last committed point becomes knot 0 of the next interval.
     */
    void FinalizeInterval() {
        if (ls_knot_count_ >= 2) {
            LSSolveAndUpdate();
        }

        // Reset LS state: last committed point becomes knot 0 of next interval
        ls_interval_start_ = out_count_ - 1;
        ls_knot_count_ = 1;
        ls_diag_.assign(1, 0.0);
        ls_offdiag_.clear();
        ls_rhs_.assign(static_cast<size_t>(n_states_), 0.0);
    }

    int GetOutputCount() const { return out_count_; }

private:
    // ════════════════════════════════════════════════════════════════════
    //  Data Members
    // ════════════════════════════════════════════════════════════════════

    // ── Core ──
    int n_states_;
    double rtol_;
    double threshold_;
    bool active_;
    std::vector<double> inv_atol_;

    // ── Output (pointers to solver's arrays) ──
    std::vector<sunrealtype>* out_t_;
    std::vector<sunrealtype>* out_y_;
    std::vector<sunrealtype>* out_yp_;
    int out_count_;

    // ── Anchor (last committed point) ──
    sunrealtype anchor_t_;
    std::vector<sunrealtype> anchor_y_, anchor_yp_;
    bool has_anchor_;

    // ── Window: points since anchor that may be removed ──
    std::vector<sunrealtype> window_t_;
    std::vector<sunrealtype> window_y_;   // Flat: [n_states * window_size]
    std::vector<sunrealtype> window_yp_;  // Flat: [n_states * window_size]

    // ── Greedy: double-buffered knot error caches ──
    std::vector<sunrealtype> dk_buf0_, dpk_buf0_, dk_buf1_, dpk_buf1_;

    // ── Greedy: sentinel states (top-K by inv_atol) ──
    std::vector<int> sentinels_;
    int n_sentinels_;

    // ── LS derivative refinement ──
    //
    // Tridiagonal normal-equation system for the integral L² objective.
    // Matrix entries are state-independent O(1) per span (Gram constants).
    // RHS is accumulated per state via factored 4-FMA inner loop.
    int ls_knot_count_;           // Knots in current interval's LS system
    int ls_interval_start_;       // Output index of current interval's first knot
    std::vector<double> ls_diag_;      // Main diagonal:  K scalars
    std::vector<double> ls_offdiag_;   // Off-diagonal:   (K-1) scalars
    std::vector<double> ls_rhs_;       // Right-hand side: K * S (state-interleaved)
    std::vector<double> ls_sum_A_;     // Per-state work buffer (S doubles)
    std::vector<double> ls_sum_B_;     // Per-state work buffer (S doubles)

    // ── Hermite mass matrix G_{ij} = ∫₀¹ hᵢ(u) hⱼ(u) du ──
    // Basis order: {h₀₀, h₁₀, h₀₁, h₁₁}
    // 10 unique entries of the 4×4 symmetric matrix.
    static constexpr double kG00 = 13.0/35.0;
    static constexpr double kG01 = 11.0/210.0;
    static constexpr double kG02 = 9.0/70.0;
    static constexpr double kG03 = -13.0/420.0;
    static constexpr double kG11 = 1.0/105.0;
    static constexpr double kG12 = 13.0/420.0;
    static constexpr double kG13 = -1.0/140.0;
    static constexpr double kG22 = 13.0/35.0;
    static constexpr double kG23 = -11.0/210.0;
    static constexpr double kG33 = 1.0/105.0;

    // ════════════════════════════════════════════════════════════════════
    //  Hermite Basis: all 8 values at a normalized coordinate u
    // ════════════════════════════════════════════════════════════════════

    /**
     * @brief All 8 Hermite basis function values at a normalized coordinate.
     *
     * For a merged span of length h with u = (t - t_anchor) / h:
     *
     *   Sensitivity basis (LS φ functions):
     *     phiA  = h₁₀(u)·h = (u³ - 2u² + u)·h      sensitivity to anchor y'
     *     dphiA = h₁₀'(u)  = 3u² - 4u + 1            its derivative
     *     phiB  = h₁₁(u)·h = (u³ - u²)·h             sensitivity to candidate y'
     *     dphiB = h₁₁'(u)  = 3u² - 2u                 its derivative
     *
     *   Value interpolation basis (for knot error d_k = y_k - H(t_k)):
     *     cv0 = h₀₀(u) = 2u³ - 3u² + 1               anchor value weight
     *     cv2 = h₀₁(u) = -2u³ + 3u²                   candidate value weight
     *     cd0 = h₀₀'(u)/h = 6(u²-u)/h                 anchor value deriv weight
     *     cd2 = h₀₁'(u)/h = -6(u²-u)/h = -cd0         candidate value deriv weight
     *
     * Note: cv1 = phiA, cd1 = dphiA, cv3 = phiB, cd3 = dphiB.
     */
    struct MergedBasis {
        double phiA, dphiA, phiB, dphiB;
        double cv0, cv2, cd0, cd2;

        static MergedBasis AtInterior(double u, double h, double inv_h) {
            const double u2 = u * u, u3 = u2 * u;
            return {
                (u3 - 2.0*u2 + u) * h,       // phiA
                3.0*u2 - 4.0*u + 1.0,         // dphiA
                (u3 - u2) * h,                 // phiB
                3.0*u2 - 2.0*u,                // dphiB
                2.0*u3 - 3.0*u2 + 1.0,         // cv0
                -2.0*u3 + 3.0*u2,              // cv2
                6.0*(u2 - u) * inv_h,           // cd0
                -6.0*(u2 - u) * inv_h           // cd2
            };
        }

        static constexpr MergedBasis Anchor()    { return {0,1,0,0, 1,0,0,0}; }
        static constexpr MergedBasis Candidate() { return {0,0,0,1, 0,1,0,0}; }
    };

    // ════════════════════════════════════════════════════════════════════
    //  Point Management
    // ════════════════════════════════════════════════════════════════════

    void CommitPoint(sunrealtype t, const sunrealtype* y, const sunrealtype* yp) {
        LSEnsureArrays();
        ++ls_knot_count_;
        
        out_t_->push_back(t);
        out_y_->insert(out_y_->end(), y, y + n_states_);
        out_yp_->insert(out_yp_->end(), yp, yp + n_states_);
        
        ++out_count_;
    }

    void SetAnchor(sunrealtype t, const sunrealtype* y, const sunrealtype* yp) {
        anchor_t_ = t;
        anchor_y_.assign(y, y + n_states_);
        anchor_yp_.assign(yp, yp + n_states_);
        has_anchor_ = true;

        window_t_.clear();
        window_y_.clear();
        window_yp_.clear();
    }

    void AddToWindow(sunrealtype t, const sunrealtype* y, const sunrealtype* yp) {
        window_t_.push_back(t);
        window_y_.insert(window_y_.end(), y, y + n_states_);
        window_yp_.insert(window_yp_.end(), yp, yp + n_states_);
    }

    /**
     * @brief Commit the current candidate (window.back()) after accumulating
     *        its LS span contributions. Does NOT clear the window.
     *
     * Extracts the repeated pattern: LSAccumulateSpan + CommitPoint for the
     * last point in the window. After this call, window data is still valid
     * (only SetAnchor clears it).
     */
    void FlushCandidate() {
        const size_t last = window_t_.size() - 1;
        LSAccumulateSpan(last);
        CommitPoint(window_t_[last], &window_y_[last * n_states_],
                    &window_yp_[last * n_states_]);
    }

    // ════════════════════════════════════════════════════════════════════
    //  Greedy Merge Check (Bernstein-certified, three-level hierarchy)
    // ════════════════════════════════════════════════════════════════════

    /**
     * @brief Compute knot errors d_k = y_k - H_merged(t_k) and d'_k via
     *        numerically stable Hermite basis evaluation in normalized coords.
     *
     * @param k       Window index of the knot (window_y_[k*n], window_yp_[k*n])
     * @param inv_hm  1 / h_merged
     * @param h_m     h_merged = t_new - t_anchor
     * @param ya,ypa  Anchor state/derivative arrays (n_states)
     * @param yn,ypn  New-point state/derivative arrays (n_states)
     * @param dk,dpk  [out] Knot value and derivative errors (n_states)
     */
    inline void ComputeKnotErrors(
            size_t k, sunrealtype inv_hm, sunrealtype h_m,
            const sunrealtype* __restrict__ ya,
            const sunrealtype* __restrict__ ypa,
            const sunrealtype* __restrict__ yn,
            const sunrealtype* __restrict__ ypn,
            sunrealtype* __restrict__ dk,
            sunrealtype* __restrict__ dpk) const
    {
        const int n = n_states_;
        const double u = (window_t_[k] - anchor_t_) * inv_hm;
        const MergedBasis B = MergedBasis::AtInterior(u, h_m, inv_hm);

        const sunrealtype* __restrict__ yk  = &window_y_[k * n];
        const sunrealtype* __restrict__ ypk = &window_yp_[k * n];

        #pragma omp simd
        for (int j = 0; j < n; ++j) {
            const sunrealtype hm_val = B.cv0*ya[j] + B.phiA*ypa[j]
                                     + B.cv2*yn[j] + B.phiB*ypn[j];
            const sunrealtype hm_deriv = B.cd0*ya[j] + B.dphiA*ypa[j]
                                       + B.cd2*yn[j] + B.dphiB*ypn[j];
            dk[j]  = yk[j]  - hm_val;
            dpk[j] = ypk[j] - hm_deriv;
        }
    }

    /**
     * @brief Sentinel pre-check: O(n_sentinels) lower bound on Level 1 sum.
     *
     * Checks the top-32 states (by inv_atol) for a certified lower bound.
     * If sentinels alone exceed threshold, the full sum certainly does too.
     * Valid because partial sum <= full sum.
     *
     * @return true if sentinels predict rejection (skip Level 1, go to Level 2)
     */
    inline bool CheckSentinels(
            const sunrealtype* __restrict__ dk_left,
            const sunrealtype* __restrict__ dpk_left,
            sunrealtype h_third) const
    {
        const int* __restrict__ sent = sentinels_.data();
        const double* __restrict__ inv_atol = inv_atol_.data();

        double sentinel_sum = 0.0;
        for (int s = 0; s < n_sentinels_; ++s) {
            const int j = sent[s];
            const sunrealtype abs_b0 = std::fabs(dk_left[j]);
            const sunrealtype abs_b1 = std::fabs(dk_left[j] + dpk_left[j] * h_third);
            const sunrealtype err = std::fmax(abs_b0, abs_b1);
            const sunrealtype e = err * inv_atol[j];
            sentinel_sum += e * e;
        }
        return sentinel_sum > threshold_;
    }

    /**
     * @brief Level 1: Conservative Bernstein bound (division-free, SIMD).
     *
     * Uses inv_atol >= inv_w as upper bound on the WRMS weight.
     * First/last sub-intervals check only 2 control points (anchor/new-point
     * error is exactly zero).
     *
     * @return Sum of squared weighted Bernstein bounds across all states.
     */
    inline double CheckLevel1(
            const sunrealtype* __restrict__ dk_left,
            const sunrealtype* __restrict__ dpk_left,
            const sunrealtype* __restrict__ dk_right,
            const sunrealtype* __restrict__ dpk_right,
            sunrealtype h_third, bool is_first, bool is_last) const
    {
        const int n = n_states_;
        const double* __restrict__ inv_atol = inv_atol_.data();
        double sum_sq = 0.0;

        if (is_first) {
            // Anchor error is zero: beta0 = beta1 = 0
            #pragma omp simd reduction(+:sum_sq)
            for (int j = 0; j < n; ++j) {
                const sunrealtype abs_b2 = std::fabs(dk_right[j] - dpk_right[j] * h_third);
                const sunrealtype abs_b3 = std::fabs(dk_right[j]);
                const sunrealtype err = std::fmax(abs_b2, abs_b3);
                const sunrealtype e = err * inv_atol[j];
                sum_sq += e * e;
            }
        } else if (is_last) {
            // New-point error is zero: beta2 = beta3 = 0
            #pragma omp simd reduction(+:sum_sq)
            for (int j = 0; j < n; ++j) {
                const sunrealtype abs_b0 = std::fabs(dk_left[j]);
                const sunrealtype abs_b1 = std::fabs(dk_left[j] + dpk_left[j] * h_third);
                const sunrealtype err = std::fmax(abs_b0, abs_b1);
                const sunrealtype e = err * inv_atol[j];
                sum_sq += e * e;
            }
        } else {
            // General: all 4 Bernstein control points
            #pragma omp simd reduction(+:sum_sq)
            for (int j = 0; j < n; ++j) {
                const sunrealtype abs_b0 = std::fabs(dk_left[j]);
                const sunrealtype abs_b1 = std::fabs(dk_left[j] + dpk_left[j] * h_third);
                const sunrealtype abs_b2 = std::fabs(dk_right[j] - dpk_right[j] * h_third);
                const sunrealtype abs_b3 = std::fabs(dk_right[j]);
                const sunrealtype err = std::fmax(std::fmax(abs_b0, abs_b1),
                                                  std::fmax(abs_b2, abs_b3));
                const sunrealtype e = err * inv_atol[j];
                sum_sq += e * e;
            }
        }
        return sum_sq;
    }

    /**
     * @brief Level 2: Bernstein bound with exact WRMS weight (rare path).
     *
     * Uses fmin(|y_left|, |y_right|) for conservative weight certification.
     * Only entered when Level 1's conservative bound exceeds the threshold.
     *
     * @return Sum of squared WRMS-weighted Bernstein bounds across all states.
     */
    inline double CheckLevel2(
            const sunrealtype* __restrict__ dk_left,
            const sunrealtype* __restrict__ dpk_left,
            const sunrealtype* __restrict__ dk_right,
            const sunrealtype* __restrict__ dpk_right,
            sunrealtype h_third,
            const sunrealtype* __restrict__ y_left,
            const sunrealtype* __restrict__ y_right) const
    {
        const int n = n_states_;
        const double* __restrict__ inv_atol = inv_atol_.data();
        const double rtol = rtol_;
        double sum_sq = 0.0;

        #pragma omp simd reduction(+:sum_sq)
        for (int j = 0; j < n; ++j) {
            const sunrealtype abs_b0 = std::fabs(dk_left[j]);
            const sunrealtype abs_b1 = std::fabs(dk_left[j] + dpk_left[j] * h_third);
            const sunrealtype abs_b2 = std::fabs(dk_right[j] - dpk_right[j] * h_third);
            const sunrealtype abs_b3 = std::fabs(dk_right[j]);
            const sunrealtype err = std::fmax(std::fmax(abs_b0, abs_b1),
                                              std::fmax(abs_b2, abs_b3));
            const sunrealtype ymin = std::fmin(std::fabs(y_left[j]),
                                               std::fabs(y_right[j]));
            const sunrealtype inv_w = inv_atol[j] / (1.0 + rtol * ymin * inv_atol[j]);
            const sunrealtype e = err * inv_w;
            sum_sq += e * e;
        }
        return sum_sq;
    }

    /**
     * @brief Level 3: De Casteljau midpoint subdivision tightening (very rare).
     *
     * Splits each Bernstein curve at the midpoint, producing 8 refined control
     * points whose convex hull is ~4x tighter. Uses exact WRMS weight.
     *
     * @return Sum of squared WRMS-weighted refined Bernstein bounds.
     */
    inline double CheckLevel3(
            const sunrealtype* __restrict__ dk_left,
            const sunrealtype* __restrict__ dpk_left,
            const sunrealtype* __restrict__ dk_right,
            const sunrealtype* __restrict__ dpk_right,
            sunrealtype h_third,
            const sunrealtype* __restrict__ y_left,
            const sunrealtype* __restrict__ y_right) const
    {
        const int n = n_states_;
        const double* __restrict__ inv_atol = inv_atol_.data();
        const double rtol = rtol_;
        double sum_sq = 0.0;

        #pragma omp simd reduction(+:sum_sq)
        for (int j = 0; j < n; ++j) {
            const sunrealtype B0 = dk_left[j];
            const sunrealtype B1 = dk_left[j] + dpk_left[j] * h_third;
            const sunrealtype B2 = dk_right[j] - dpk_right[j] * h_third;
            const sunrealtype B3 = dk_right[j];

            const sunrealtype L1 = 0.5 * (B0 + B1);
            const sunrealtype L2 = 0.25 * (B0 + 2.0 * B1 + B2);
            const sunrealtype L3 = 0.125 * (B0 + 3.0 * (B1 + B2) + B3);

            const sunrealtype R1 = 0.25 * (B1 + 2.0 * B2 + B3);
            const sunrealtype R2 = 0.5 * (B2 + B3);

            const sunrealtype max_left = std::fmax(
                std::fmax(std::fabs(B0), std::fabs(L1)),
                std::fmax(std::fabs(L2), std::fabs(L3)));
            const sunrealtype max_right = std::fmax(
                std::fmax(std::fabs(L3), std::fabs(R1)),
                std::fmax(std::fabs(R2), std::fabs(B3)));
            const sunrealtype err = std::fmax(max_left, max_right);

            const sunrealtype ymin = std::fmin(std::fabs(y_left[j]),
                                               std::fabs(y_right[j]));
            const sunrealtype inv_w = inv_atol[j] / (1.0 + rtol * ymin * inv_atol[j]);
            const sunrealtype e = err * inv_w;
            sum_sq += e * e;
        }
        return sum_sq;
    }

    /**
     * @brief Three-level Bernstein-certified merge check with sentinel early exit.
     *
     * For the error polynomial d(s) on sub-interval [t_k, t_{k+1}] with step h:
     *   beta0 = d_k,  beta1 = d_k + d'_k*h/3,
     *   beta2 = d_{k+1} - d'_{k+1}*h/3,  beta3 = d_{k+1}
     *
     * Convex hull property: max|d(s)| <= max(|beta_i|) for s in [0, h]
     *
     * SENTINEL EARLY EXIT:
     *   Before the full O(n) SIMD loop, check sentinel states (top-32 by
     *   inv_atol). Their partial WRMS sum is a certified lower bound on the
     *   full sum. If sentinels alone exceed threshold, skip O(n) Level 1
     *   entirely and proceed to Level 2. Zero risk: subset sum <= full sum.
     *
     * Knot errors (d_k, d'_k) are evaluated ONCE per knot using numerically
     * stable Hermite basis functions in normalized coordinates, then shared
     * between adjacent sub-intervals via double buffering.
     */
    bool CanMergeWindow(sunrealtype t_new, const sunrealtype* y_new, const sunrealtype* yp_new) {
        const int n = n_states_;

        const sunrealtype h_merged = t_new - anchor_t_;
        const sunrealtype inv_hm = 1.0 / h_merged;

        const sunrealtype* __restrict__ ya = anchor_y_.data();
        const sunrealtype* __restrict__ ypa = anchor_yp_.data();

        // Double-buffered knot error caches
        sunrealtype* __restrict__ dk_left = dk_buf0_.data();
        sunrealtype* __restrict__ dpk_left = dpk_buf0_.data();
        sunrealtype* __restrict__ dk_right = dk_buf1_.data();
        sunrealtype* __restrict__ dpk_right = dpk_buf1_.data();

        const size_t window_size = window_t_.size();

        for (size_t k = 0; k <= window_size; ++k) {
            const bool is_first = (k == 0);
            const sunrealtype t_left = is_first ? anchor_t_ : window_t_[k - 1];
            const sunrealtype* __restrict__ y_left = is_first ? ya : &window_y_[(k - 1) * n];

            const bool is_last = (k == window_size);
            const sunrealtype t_right = is_last ? t_new : window_t_[k];
            const sunrealtype* __restrict__ y_right = is_last ? y_new : &window_y_[k * n];

            const sunrealtype h_sub = t_right - t_left;

            // Skip degenerate sub-intervals
            if (h_sub <= 0.0) {
                if (!is_last) {
                    std::swap(dk_left, dk_right);
                    std::swap(dpk_left, dpk_right);
                }
                continue;
            }

            const sunrealtype h_third = h_sub / 3.0;

            // ── Sentinel pre-check ──
            bool skip_level1 = false;
            if (!is_first && n_sentinels_ > 0) {
                if (CheckSentinels(dk_left, dpk_left, h_third)) {
                    if (!is_last) {
                        ComputeKnotErrors(k, inv_hm, h_merged,
                                          ya, ypa, y_new, yp_new,
                                          dk_right, dpk_right);
                    }
                    skip_level1 = true;
                }
            }

            if (!skip_level1) {
                if (!is_last) {
                    ComputeKnotErrors(k, inv_hm, h_merged,
                                      ya, ypa, y_new, yp_new,
                                      dk_right, dpk_right);
                }

                // ── Level 1: Conservative Bernstein (division-free) ──
                const double l1_sum = CheckLevel1(dk_left, dpk_left,
                                                  dk_right, dpk_right,
                                                  h_third, is_first, is_last);
                if (l1_sum <= threshold_) {
                    std::swap(dk_left, dk_right);
                    std::swap(dpk_left, dpk_right);
                    continue;
                }
            }

            // ── Level 2: Bernstein with exact WRMS weight (rare path) ──
            // Deferred memset: only zero the endpoint buffers on this rare path.
            if (is_first) {
                std::memset(dk_left, 0, n * sizeof(sunrealtype));
                std::memset(dpk_left, 0, n * sizeof(sunrealtype));
            }
            if (is_last) {
                std::memset(dk_right, 0, n * sizeof(sunrealtype));
                std::memset(dpk_right, 0, n * sizeof(sunrealtype));
            }

            const double l2_sum = CheckLevel2(dk_left, dpk_left,
                                              dk_right, dpk_right,
                                              h_third, y_left, y_right);

            if (l2_sum > threshold_) {
                // ── Level 3: De Casteljau midpoint subdivision ──
                if (l2_sum > 9.0 * threshold_) {
                    return false;
                }
                const double l3_sum = CheckLevel3(dk_left, dpk_left,
                                                  dk_right, dpk_right,
                                                  h_third, y_left, y_right);
                if (l3_sum > threshold_) {
                    return false;
                }
            }

            std::swap(dk_left, dk_right);
            std::swap(dpk_left, dpk_right);
        }

        return true;
    }

    // ════════════════════════════════════════════════════════════════════
    //  LS Derivative Refinement: Streaming Accumulation + Thomas Solve
    // ════════════════════════════════════════════════════════════════════

    /**
     * @brief Ensure LS arrays are sized for the current knot count.
     *
     * Resizes diag, offdiag, and rhs arrays to accommodate ls_knot_count_ + 1
     * entries. New entries are zero-initialized. Safe to call multiple times
     * (resize is a no-op when already at the correct size).
     */
    void LSEnsureArrays() {
        const size_t kp1 = static_cast<size_t>(ls_knot_count_ + 1);
        ls_diag_.resize(kp1, 0.0);
        ls_rhs_.resize(kp1 * n_states_, 0.0);
        if (ls_knot_count_ > 0) {
            ls_offdiag_.resize(static_cast<size_t>(ls_knot_count_), 0.0);
        }
    }

    /**
     * @brief Accumulate LS matrix entries (O(1) per span, closed-form).
     *
     * Adds the Hermite mass matrix contributions for a span of length h_span:
     *   diag[k-1] += h³/105,  diag[k] += h³/105,  offdiag[k-1] += -h³/140
     *
     * Used for both full spans (via LSAccumulateSpan) and empty spans with
     * no interior points (e.g., candidate→breakpoint or anchor→breakpoint).
     *
     * @param h_span  Length of the span
     */
    void LSAccumulateMatrix(sunrealtype h_span) {
        if (ls_knot_count_ == 0 || h_span <= 0.0) return;
        LSEnsureArrays();
        const int k = ls_knot_count_;
        const double h3 = h_span * h_span * h_span;
        ls_diag_[k - 1]    += h3 * kG11;   // h³/105
        ls_diag_[k]        += h3 * kG33;   // h³/105
        ls_offdiag_[k - 1] += h3 * kG13;   // -h³/140
    }

    /**
     * @brief Accumulate LS matrix + RHS for a closing span.
     *
     * Thin dispatcher: accumulates the closed-form matrix entries, then
     * (if there are interior points) the factored RHS contributions.
     *
     * Called just BEFORE CommitPoint for window[last].
     *
     * @param last  Index of the candidate in the window (window.back())
     */
    void LSAccumulateSpan(size_t last) {
        if (ls_knot_count_ == 0) return;
        const sunrealtype h = window_t_[last] - anchor_t_;
        LSAccumulateMatrix(h);
        if (last > 0 && h > 0.0) {
            LSAccumulateRHS(last, h);
        }
    }

    /**
     * @brief Accumulate LS RHS via factored sliding-window accumulation.
     *
     * Processes interior points p = 0..last-1 in a single pass with a
     * prev → curr → next sliding window of MergedBasis values.
     *
     * For each interior point, computes Gram-weighted per-node coefficients
     * from the left [prev,p] and right [p,next] sub-intervals, then:
     *   - Updates scalar accumulators for constant endpoint correction
     *   - Accumulates per-state sums via 4 FMA (the dominant cost)
     *
     * At span closure, a single finalization pass combines the per-state
     * sums with the scalar corrections to form the final RHS entries.
     *
     * @param last  Index of the candidate (window.back())
     * @param h     Merged span length (t_candidate - t_anchor, must be > 0)
     */
    void LSAccumulateRHS(size_t last, sunrealtype h) {
        const int S = n_states_;
        const int k = ls_knot_count_;
        const int km1 = k - 1;
        const sunrealtype inv_h = 1.0 / h;

        const sunrealtype* __restrict__ ya  = anchor_y_.data();
        const sunrealtype* __restrict__ ypa = anchor_yp_.data();
        const sunrealtype* __restrict__ yr  = &window_y_[last * S];
        const sunrealtype* __restrict__ ypr = &window_yp_[last * S];

        double* __restrict__ r_left  = &ls_rhs_[static_cast<size_t>(km1) * S];
        double* __restrict__ r_right = &ls_rhs_[static_cast<size_t>(k) * S];

        // Zero per-state work buffers
        double* __restrict__ sum_A = ls_sum_A_.data();
        double* __restrict__ sum_B = ls_sum_B_.data();
        std::memset(sum_A, 0, static_cast<size_t>(S) * sizeof(double));
        std::memset(sum_B, 0, static_cast<size_t>(S) * sizeof(double));

        // Scalar accumulators for endpoint correction (hoisted out of state loop)
        double s_ya_A = 0, s_ypa_A = 0, s_yr_A = 0, s_ypr_A = 0;
        double s_ya_B = 0, s_ypa_B = 0, s_yr_B = 0, s_ypr_B = 0;

        // ── Sliding window: prev → curr → next ──
        MergedBasis prev = MergedBasis::Anchor();
        double t_prev = anchor_t_;

        const double u0 = (window_t_[0] - anchor_t_) * inv_h;
        MergedBasis curr = MergedBasis::AtInterior(u0, h, inv_h);

        for (size_t p = 0; p < last; ++p) {
            // Lookahead: compute next point's basis
            const double t_next = (p + 1 < last) ? window_t_[p + 1] : window_t_[last];
            const MergedBasis next = (p + 1 < last)
                ? MergedBasis::AtInterior((t_next - anchor_t_) * inv_h, h, inv_h)
                : MergedBasis::Candidate();

            // Sub-interval lengths
            const double h_L = window_t_[p] - t_prev;
            const double h_R = t_next - window_t_[p];

            // ── Left sub-interval [prev, p]: Gram-weighted rows 2,3 ──
            const double aL0 = prev.phiA,  aL1 = h_L * prev.dphiA;
            const double aL2 = curr.phiA,  aL3 = h_L * curr.dphiA;
            const double qAL2 = kG02*aL0 + kG12*aL1 + kG22*aL2 + kG23*aL3;
            const double qAL3 = kG03*aL0 + kG13*aL1 + kG23*aL2 + kG33*aL3;

            const double bL0 = prev.phiB,  bL1 = h_L * prev.dphiB;
            const double bL2 = curr.phiB,  bL3 = h_L * curr.dphiB;
            const double qBL2 = kG02*bL0 + kG12*bL1 + kG22*bL2 + kG23*bL3;
            const double qBL3 = kG03*bL0 + kG13*bL1 + kG23*bL2 + kG33*bL3;

            // ── Right sub-interval [p, next]: Gram-weighted rows 0,1 ──
            const double aR0 = curr.phiA,  aR1 = h_R * curr.dphiA;
            const double aR2 = next.phiA,  aR3 = h_R * next.dphiA;
            const double qAR0 = kG00*aR0 + kG01*aR1 + kG02*aR2 + kG03*aR3;
            const double qAR1 = kG01*aR0 + kG11*aR1 + kG12*aR2 + kG13*aR3;

            const double bR0 = curr.phiB,  bR1 = h_R * curr.dphiB;
            const double bR2 = next.phiB,  bR3 = h_R * next.dphiB;
            const double qBR0 = kG00*bR0 + kG01*bR1 + kG02*bR2 + kG03*bR3;
            const double qBR1 = kG01*bR0 + kG11*bR1 + kG12*bR2 + kG13*bR3;

            // ── Combined per-node coefficients ──
            const double c_d_A  = h_L * qAL2 + h_R * qAR0;
            const double c_dp_A = h_L * h_L * qAL3 + h_R * h_R * qAR1;
            const double c_d_B  = h_L * qBL2 + h_R * qBR0;
            const double c_dp_B = h_L * h_L * qBL3 + h_R * h_R * qBR1;

            // ── Scalar accumulators: absorb constant endpoint data ──
            s_ya_A  += c_d_A * curr.cv0 + c_dp_A * curr.cd0;
            s_ypa_A += c_d_A * curr.phiA + c_dp_A * curr.dphiA;  // cv1=phiA, cd1=dphiA
            s_yr_A  += c_d_A * curr.cv2 + c_dp_A * curr.cd2;
            s_ypr_A += c_d_A * curr.phiB + c_dp_A * curr.dphiB;  // cv3=phiB, cd3=dphiB

            s_ya_B  += c_d_B * curr.cv0 + c_dp_B * curr.cd0;
            s_ypa_B += c_d_B * curr.phiA + c_dp_B * curr.dphiA;
            s_yr_B  += c_d_B * curr.cv2 + c_dp_B * curr.cd2;
            s_ypr_B += c_d_B * curr.phiB + c_dp_B * curr.dphiB;

            // ── Per-state accumulation (4 FMA per state) ──
            const sunrealtype* __restrict__ y_p  = &window_y_[p * S];
            const sunrealtype* __restrict__ yp_p = &window_yp_[p * S];

            #pragma omp simd
            for (int s = 0; s < S; ++s) {
                sum_A[s] += c_d_A * y_p[s] + c_dp_A * yp_p[s];
                sum_B[s] += c_d_B * y_p[s] + c_dp_B * yp_p[s];
            }

            // ── Advance sliding window ──
            t_prev = window_t_[p];
            prev = curr;
            curr = next;
        }

        // ── Finalize: RHS = accumulated sums - scalar corrections × endpoints ──
        #pragma omp simd
        for (int s = 0; s < S; ++s) {
            r_left[s]  += sum_A[s] - s_ya_A * ya[s] - s_ypa_A * ypa[s]
                                   - s_yr_A * yr[s] - s_ypr_A * ypr[s];
            r_right[s] += sum_B[s] - s_ya_B * ya[s] - s_ypa_B * ypa[s]
                                   - s_yr_B * yr[s] - s_ypr_B * ypr[s];
        }
    }

    /**
     * @brief Solve the tri-diagonal LS system and update yp.
     *
     * Thomas algorithm for the SPD tri-diagonal system arising from the
     * integral L² objective.
     *
     * Because the normal-equation matrix is state-independent, the Thomas
     * factorization is O(K) SCALAR operations. The scalar multiplier m[k]
     * is then broadcast to all S states during the forward RHS sweep and
     * back-substitution (O(K*S)).
     *
     * The integral objective guarantees the matrix is ALWAYS SPD:
     *   - diag[k] = h_left³/105 + h_right³/105 > 0  for all k
     *   - Strict diagonal dominance: 1/105 > 1/140
     *   - No zero-diagonal checks needed
     *   - Completely branch-free inner loops
     */
    void LSSolveAndUpdate() {
        const int K = ls_knot_count_;
        const int S = n_states_;

        if (K < 2) return;

        // For non-first intervals, knot 0 was already updated by the previous
        // interval's LS solve (as its last knot). Fix δ_0 = 0 to prevent
        // double-update: zero row 0 and decouple it from knot 1.
        if (ls_interval_start_ > 0) {
            ls_diag_[0] = 1.0;
            ls_offdiag_[0] = 0.0;
            std::memset(&ls_rhs_[0], 0, static_cast<size_t>(S) * sizeof(double));
        }

        // ── 1. Thomas factorization + forward RHS sweep ──
        for (int k = 1; k < K; ++k) {
            const double m = ls_offdiag_[k - 1] / ls_diag_[k - 1];
            ls_diag_[k] -= m * ls_offdiag_[k - 1];

            const double* __restrict__ r_prev = &ls_rhs_[static_cast<size_t>(k - 1) * S];
            double* __restrict__ r_curr       = &ls_rhs_[static_cast<size_t>(k) * S];

            #pragma omp simd
            for (int s = 0; s < S; ++s) {
                r_curr[s] -= m * r_prev[s];
            }
        }

        // ── 2. Fused back-substitution + yp update ──
        sunrealtype* yp_data = out_yp_->data()
                             + static_cast<size_t>(ls_interval_start_) * S;

        // Last knot
        {
            const double d_inv = 1.0 / ls_diag_[K - 1];
            double* __restrict__ x_last = &ls_rhs_[static_cast<size_t>(K - 1) * S];
            sunrealtype* __restrict__ yp_k = &yp_data[static_cast<size_t>(K - 1) * S];

            #pragma omp simd
            for (int s = 0; s < S; ++s) {
                x_last[s] *= d_inv;
                yp_k[s] += static_cast<sunrealtype>(x_last[s]);
            }
        }

        // Remaining knots (backward)
        for (int k = K - 2; k >= 0; --k) {
            const double off_k = ls_offdiag_[k];
            const double d_inv = 1.0 / ls_diag_[k];
            double* __restrict__ x_curr       = &ls_rhs_[static_cast<size_t>(k) * S];
            const double* __restrict__ x_next = &ls_rhs_[static_cast<size_t>(k + 1) * S];
            sunrealtype* __restrict__ yp_k    = &yp_data[static_cast<size_t>(k) * S];

            #pragma omp simd
            for (int s = 0; s < S; ++s) {
                x_curr[s] = (x_curr[s] - off_k * x_next[s]) * d_inv;
                yp_k[s] += static_cast<sunrealtype>(x_curr[s]);
            }
        }
    }
};

#endif // PYBAMM_KNOT_REDUCER_HPP
