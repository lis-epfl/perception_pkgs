/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2018-2019 Kevin Eckenhoff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include <atomic>
#include "StateHelper.h"

#include "state/State.h"

#include "feat/FeatureDatabase.h"
#include "feat/FeatureHelper.h"
#include "types/Landmark.h"
#include "utils/colors.h"
#include "utils/print.h"

#include <boost/math/distributions/chi_squared.hpp>
#include <omp.h>
#include <cstring>
#include <algorithm>
#include <numeric>
#include "utils/vprof.h"
#include "state/ekf_cuda.h"
#include "update/UpdaterHelper.h"
#include "update/PreJac.h"

using namespace ov_core;
using namespace ov_type;
using namespace ov_msckf;

void StateHelper::EKFPropagation(std::shared_ptr<State> state, const std::vector<std::shared_ptr<Type>> &order_NEW,
                                 const std::vector<std::shared_ptr<Type>> &order_OLD, const Eigen::MatrixXd &Phi,
                                 const Eigen::MatrixXd &Q) {
  ov_msckf::g_state_epoch.fetch_add(1, std::memory_order_relaxed); // OV_PREJAC cache invalidation

  // We need at least one old and new variable
  if (order_NEW.empty() || order_OLD.empty()) {
    PRINT_ERROR(RED "StateHelper::EKFPropagation() - Called with empty variable arrays!\n" RESET);
    std::exit(EXIT_FAILURE);
  }

  // Loop through our Phi order and ensure that they are continuous in memory
  int size_order_NEW = order_NEW.at(0)->size();
  for (size_t i = 0; i < order_NEW.size() - 1; i++) {
    if (order_NEW.at(i)->id() + order_NEW.at(i)->size() != order_NEW.at(i + 1)->id()) {
      PRINT_ERROR(RED "StateHelper::EKFPropagation() - Called with non-contiguous state elements!\n" RESET);
      PRINT_ERROR(
          RED "StateHelper::EKFPropagation() - This code only support a state transition which is in the same order as the state\n" RESET);
      std::exit(EXIT_FAILURE);
    }
    size_order_NEW += order_NEW.at(i + 1)->size();
  }

  // Size of the old phi matrix
  int size_order_OLD = order_OLD.at(0)->size();
  for (size_t i = 0; i < order_OLD.size() - 1; i++) {
    size_order_OLD += order_OLD.at(i + 1)->size();
  }

  // Assert that we have correct sizes
  assert(size_order_NEW == Phi.rows());
  assert(size_order_OLD == Phi.cols());
  assert(size_order_NEW == Q.cols());
  assert(size_order_NEW == Q.rows());

  // Get the location in small phi for each measuring variable
  int current_it = 0;
  std::vector<int> Phi_id;
  for (const auto &var : order_OLD) {
    Phi_id.push_back(current_it);
    current_it += var->size();
  }

  // Loop through all our old states and get the state transition times it
  // Cov_PhiT = [ Pxx ] [ Phi' ]'
  Eigen::MatrixXd Cov_PhiT = Eigen::MatrixXd::Zero(state->_Cov.rows(), Phi.rows());
  for (size_t i = 0; i < order_OLD.size(); i++) {
    std::shared_ptr<Type> var = order_OLD.at(i);
    Cov_PhiT.noalias() +=
        state->_Cov.block(0, var->id(), state->_Cov.rows(), var->size()) * Phi.block(0, Phi_id[i], Phi.rows(), var->size()).transpose();
  }

  // Get Phi_NEW*Covariance*Phi_NEW^t + Q
  Eigen::MatrixXd Phi_Cov_PhiT = Q.selfadjointView<Eigen::Upper>();
  for (size_t i = 0; i < order_OLD.size(); i++) {
    std::shared_ptr<Type> var = order_OLD.at(i);
    Phi_Cov_PhiT.noalias() += Phi.block(0, Phi_id[i], Phi.rows(), var->size()) * Cov_PhiT.block(var->id(), 0, var->size(), Phi.rows());
  }

  // We are good to go!
  int start_id = order_NEW.at(0)->id();
  int phi_size = Phi.rows();
  int total_size = state->_Cov.rows();
  state->_Cov.block(start_id, 0, phi_size, total_size) = Cov_PhiT.transpose();
  state->_Cov.block(0, start_id, total_size, phi_size) = Cov_PhiT;
  state->_Cov.block(start_id, start_id, phi_size, phi_size) = Phi_Cov_PhiT;

  // We should check if we are not positive semi-definitate (i.e. negative diagionals is not s.p.d)
  Eigen::VectorXd diags = state->_Cov.diagonal();
  bool found_neg = false;
  for (int i = 0; i < diags.rows(); i++) {
    if (diags(i) < 0.0) {
      PRINT_WARNING(RED "StateHelper::EKFPropagation() - diagonal at %d is %.2f\n" RESET, i, diags(i));
      found_neg = true;
    }
  }
  if (found_neg) {
    std::exit(EXIT_FAILURE);
  }
}

namespace {

// Grow-only thread-local scratch. Every Eigen temporary above 128 KB goes through mmap, so the
// six multi-MB temporaries of EKFUpdate took fresh page faults on EVERY call (~3 ms/whale
// measured). Eigen's own MatrixXd::resize() does NOT fix this: it reallocates whenever the
// element COUNT changes, and m changes call to call. Hence raw buffers that only ever grow,
// viewed through Eigen::Map. Same arithmetic, same leading dimensions, bit-identical output.
inline double *ekf_scratch(std::vector<double> &buf, size_t need) {
  if (buf.size() < need) buf.resize(need);
  return buf.data();
}

// In-place tiled mirror of the upper triangle into the lower one, replacing
// `Cov = Cov.selfadjointView<Upper>()` which materialises a full N x N temporary and copies it
// back. Column-major: element (r,c) lives at cov[c*ld + r]; we set the strictly-lower entries
// from their transposes. 64x64 tiling keeps the strided reads in cache.
void symmetrize_upper_to_lower(double *cov, int N, int ld) {
  const int T = 64;
  for (int j0 = 0; j0 < N; j0 += T) {
    const int jw = std::min(T, N - j0);
    for (int i0 = j0; i0 < N; i0 += T) {
      const int ie = std::min(i0 + T, N);
      for (int c = j0; c < j0 + jw; c++) {
        const int rlo = (i0 == j0) ? c + 1 : i0;
        for (int r = rlo; r < ie; r++)
          cov[(size_t)c * ld + r] = cov[(size_t)r * ld + c];
      }
    }
  }
}

} // namespace

// Shared body of both public EKFUpdate overloads. Rp == nullptr means "R is sigma2 * I", which
// lets the isotropic callers skip materialising an m x m identity and add the scalar straight
// onto S's diagonal.
void StateHelper::ekf_update_impl(std::shared_ptr<State> state, const std::vector<std::shared_ptr<Type>> &H_order,
                                  const Eigen::MatrixXd &H, const Eigen::VectorXd &res, const Eigen::MatrixXd *Rp, double sigma2,
                                  Eigen::VectorXd *dx_out) {
  ov_msckf::g_state_epoch.fetch_add(1, std::memory_order_relaxed); // OV_PREJAC cache invalidation

  //==========================================================
  //==========================================================
  // Part of the Kalman Gain K = (P*H^T)*S^{-1} = M*S^{-1}
  VPROF("7.ekf/TOTAL");
  // dimensions of the dense linear algebra, for sizing a cuBLAS port
  VSTAT("ekf_dim/P_rows(state)", state->_Cov.rows());
  VSTAT("ekf_dim/m_rows(meas)", res.rows());
  VSTAT("ekf_dim/H_cols(involved)", H.cols());
  assert(Rp == nullptr || res.rows() == Rp->rows());
  assert(H.rows() == res.rows());
  const int N_st_ = (int)state->_Cov.rows(), m_rows_ = (int)res.rows(), H_cols_ = (int)H.cols();
  // OV_EKF_SCRATCH: Phase-0 allocation/copy elimination. Default OFF -- with it off this
  // function is byte-for-byte the previously validated code path (fresh zero-filled temporaries,
  // get_marginal_covariance, selfadjointView symmetrise, materialised R).
  static const bool scr = [] { const char *e = std::getenv("OV_EKF_SCRATCH"); return e && *e == '1'; }();
  static thread_local std::vector<double> sc_Ma_, sc_Pg_, sc_Ps_, sc_S_;
  std::vector<double> loc_Ma_, loc_Pg_, loc_S_; // per-call (mmap + zero fill), the OFF path
  auto pick = [&](std::vector<double> &sc, std::vector<double> &loc, size_t need) -> double * {
    if (scr) return ekf_scratch(sc, need);
    loc.assign(need, 0.0);
    return loc.data();
  };
  // M_a is fully overwritten by the noalias() product below, so the Zero() fill is pure waste.
  Eigen::Map<Eigen::MatrixXd> M_a(pick(sc_Ma_, loc_Ma_, (size_t)N_st_ * m_rows_), N_st_, m_rows_);

  // Get the location in small jacobian for each measuring variable
  int current_it = 0;
  std::vector<int> H_id;
  for (const auto &meas_var : H_order) {
    H_id.push_back(current_it);
    current_it += meas_var->size();
  }

  //==========================================================
  //==========================================================
  // For each active variable find its M = P*H^T
  // M_a = P[:, involved] * H^T, as ONE dense GEMM.
  // The original nested loop issued |state->_variables| x |H_order| tiny products -- roughly
  // 300 x 25 GEMMs of shape (3x6)*(6xm) -- where Eigen's per-call overhead dominates the
  // arithmetic. state->_variables tile the full covariance, so the loop was only computing
  // this product row-block by row-block. Gathering the involved columns once (a few hundred
  // KB memcpy) and doing a single GEMM is the same result with one blocked kernel.
  // Gather the involved covariance columns once; both the GPU and CPU paths need this.
  Eigen::Map<Eigen::MatrixXd> P_gathered(pick(sc_Pg_, loc_Pg_, (size_t)N_st_ * H_cols_), N_st_, H_cols_);
  { VPROF("7.ekf/a0_gather");
  for (size_t i = 0; i < H_order.size(); i++) {
    const std::shared_ptr<Type> &meas_var = H_order[i];
    P_gathered.block(0, H_id[i], P_gathered.rows(), meas_var->size()) =
        state->_Cov.block(0, meas_var->id(), state->_Cov.rows(), meas_var->size());
  } }

  // ---- fp32 GPU fast path for the whole dense block ------------------------------------
  // R is isotropic here by construction (callers build sigma_pix_sq * I), which the GPU path
  // relies on. Anything unexpected -> fall through to the CPU code below.
  if (ekf_cuda_enabled() && m_rows_ > 0 && (Rp == nullptr || (Rp->rows() == Rp->cols() && Rp->rows() > 0))) {
    const int Nst = N_st_, Hc = H_cols_, mm = m_rows_;
    bool iso = true;
    const double s2 = (Rp == nullptr) ? sigma2 : (*Rp)(0, 0);
    if (Rp != nullptr)
      for (int i = 1; i < Rp->rows() && iso; i++) if ((*Rp)(i, i) != s2) iso = false;
    if (iso) {
      VPROF("7.ekf/GPU_fp32");
      Eigen::MatrixXd P_small_g = StateHelper::get_marginal_covariance(state, H_order);
      Eigen::MatrixXf D(Nst, Nst);
      Eigen::VectorXd dxg(Nst);
      if (ekf_cuda_update(P_gathered.data(), Nst, Hc, P_small_g.data(), H.data(), mm,
                          res.data(), s2, D.data(), dxg.data())) {
        // subtract the fp32 correction from an fp64 covariance, then symmetrise
        state->_Cov.triangularView<Eigen::Upper>() -= D.cast<double>();
        state->_Cov.triangularView<Eigen::StrictlyLower>() =
            state->_Cov.transpose().triangularView<Eigen::StrictlyLower>();
        Eigen::VectorXd dg = state->_Cov.diagonal();
        for (int i = 0; i < dg.rows(); i++) {
          if (dg(i) < 0.0) {
            PRINT_WARNING(RED "StateHelper::EKFUpdate() [gpu] - diagonal at %d is %.2f\n" RESET, i, dg(i));
            std::exit(EXIT_FAILURE);
          }
        }
        if (dx_out) *dx_out = dxg;
        for (size_t i = 0; i < state->_variables.size(); i++)
          state->_variables.at(i)->update(
              dxg.block(state->_variables.at(i)->id(), 0, state->_variables.at(i)->size(), 1));
        if (state->_options.do_calib_camera_intrinsics)
          for (auto const &calib : state->_cam_intrinsics)
            state->_cam_intrinsics_cameras.at(calib.first)->set_value(calib.second->value());
        state->push_nonradial_to_cameras();
        return;
      }
    }
  }
  // ---------------------------------------------------------------------------------------
  // CPU path: the GPU either is disabled or bailed out, so build M_a here.
  // OV_DS_EKF: big M_a products go to the GPU double-single GEMM (fp64-class accuracy at
  // fp32 speed); small ones stay on Eigen where transfer overhead would dominate.
  static const long ds_min = [] { const char *e = std::getenv("OV_DS_MIN_MMAC"); return e ? atol(e) : 20L; }();
  { VPROF("7.ekf/a_M=P*Ht");
  bool done_gpu = false;
  if (ekf_ds_available() &&
      (long)state->_Cov.rows() * H.cols() * res.rows() > ds_min * 1000000L) {
    VPROF("7.ekf/a_M_dsgpu");
    done_gpu = ekf_ds_gemm_nt(P_gathered.data(), (int)P_gathered.rows(), (int)P_gathered.cols(),
                              H.data(), (int)H.rows(), M_a.data(), false);
  }
  if (!done_gpu)
    M_a.noalias() = P_gathered * H.transpose(); }
  // OV_EKF_VERIFY: does writing the GEMM into a reused, possibly-unaligned Map destination
  // change any bit versus a freshly allocated MatrixXd? (Alignment affects load/store
  // instruction selection, not accumulation order -- but assert it rather than assume it.)
  static const bool ekf_verify0 = [] { const char *e = std::getenv("OV_EKF_VERIFY"); return e && *e == '1'; }();
  if (ekf_verify0) {
    Eigen::MatrixXd Pg_ref = P_gathered;
    Eigen::MatrixXd Ma_ref = Eigen::MatrixXd::Zero(N_st_, m_rows_);
    Ma_ref.noalias() = Pg_ref * H.transpose();
    double d = (Ma_ref - Eigen::MatrixXd(M_a)).cwiseAbs().maxCoeff();
    printf("[ekfverify] M_a N=%d m=%d dmax=%.3e %s\n", N_st_, m_rows_, d, d == 0.0 ? "BITEXACT" : "DIFFER");
  }

  //==========================================================
  //==========================================================
  // Get covariance of the involved terms
  // P_small is EXACTLY the involved rows of P_gathered: P_gathered(:, H_id[i]+c) is
  // _Cov(:, var_i->id()+c), so P_small(H_id[k]+a, :) = P_gathered(var_k->id()+a, :). One
  // middleRows copy per variable (~57) instead of get_marginal_covariance's |vars|^2 (~3200)
  // 6x6 block copies plus a 0.95 MB Zero() fill. Same bytes, same values.
  Eigen::MatrixXd P_small_old_; // OFF path only
  { VPROF("7.ekf/b_marg_cov");
  if (!scr) P_small_old_ = StateHelper::get_marginal_covariance(state, H_order); }
  Eigen::Map<Eigen::MatrixXd> P_small(scr ? ekf_scratch(sc_Ps_, (size_t)H_cols_ * H_cols_) : P_small_old_.data(),
                                      H_cols_, H_cols_);
  { VPROF("7.ekf/b_marg_cov");
  if (scr)
    for (size_t i = 0; i < H_order.size(); i++) {
      const int sz = (int)H_order[i]->size();
      P_small.middleRows(H_id[i], sz) = P_gathered.middleRows(H_order[i]->id(), sz);
    } }
  // OV_EKF_VERIFY: the row-gather above must reproduce get_marginal_covariance EXACTLY (it is a
  // pure copy of the same covariance entries, just routed through P_gathered).
  static const bool ekf_verify = [] { const char *e = std::getenv("OV_EKF_VERIFY"); return e && *e == '1'; }();
  if (ekf_verify) {
    Eigen::MatrixXd Pref = StateHelper::get_marginal_covariance(state, H_order);
    double dmax = (Pref - Eigen::MatrixXd(P_small)).cwiseAbs().maxCoeff();
    printf("[ekfverify] Psmall n=%d dmax=%.3e %s\n", H_cols_, dmax, dmax == 0.0 ? "BITEXACT" : "DIFFER");
  }

  // Residual covariance S = H*Cov*H' + R
  Eigen::Map<Eigen::MatrixXd> S(pick(sc_S_, loc_S_, (size_t)m_rows_ * m_rows_), m_rows_, m_rows_);
  { VPROF("7.ekf/c_S=HPHt+R");
  S.triangularView<Eigen::Upper>() = H * P_small * H.transpose();
  // R = sigma2 * I adds sigma2 on the diagonal and exact 0.0 off it, so the scalar form is
  // bit-identical to the materialised identity it replaces.
  if (Rp != nullptr) S.triangularView<Eigen::Upper>() += *Rp;
  else if (scr) S.diagonal().array() += sigma2;
  else { Eigen::MatrixXd Rmat = sigma2 * Eigen::MatrixXd::Identity(m_rows_, m_rows_);
         S.triangularView<Eigen::Upper>() += Rmat; } }
  if (ekf_verify0) {
    // S built the old way: fresh temporary + materialised R, upper triangle only.
    Eigen::MatrixXd Ps_ref = P_small, S_ref(m_rows_, m_rows_);
    S_ref.triangularView<Eigen::Upper>() = H * Ps_ref * H.transpose();
    Eigen::MatrixXd Rm = (Rp != nullptr) ? *Rp : Eigen::MatrixXd(sigma2 * Eigen::MatrixXd::Identity(m_rows_, m_rows_));
    S_ref.triangularView<Eigen::Upper>() += Rm;
    double d = (S_ref.triangularView<Eigen::Upper>().toDenseMatrix() -
                Eigen::MatrixXd(S).triangularView<Eigen::Upper>().toDenseMatrix()).cwiseAbs().maxCoeff();
    printf("[ekfverify] S m=%d dmax=%.3e %s\n", m_rows_, d, d == 0.0 ? "BITEXACT" : "DIFFER");
  }
  // Eigen::MatrixXd S = H * P_small * H.transpose() + R;

  // OV_CHOL_DOWN: Cholesky-factored downdate. With S = U^T U (U upper) and V = M_a U^{-1},
  // the optimal-gain downdate is EXACTLY  P -= V V^T  and  dx = V (U^{-T} res)  -- the K
  // matrix is never formed. Same math as the Sinv+GEMM path below, but:
  //   * half the downdate flops (N^2 m vs 2 N^2 m: SYRK vs full GEMM), threaded by column
  //     blocks (Eigen's rankUpdate is single-threaded, hence the hand blocking);
  //   * the N x m^2 K GEMM and the m^3 explicit Sinv are replaced by an N x m^2 / 2
  //     triangular solve, threaded by row blocks;
  //   * V V^T is symmetric PSD by construction, so the downdate cannot break symmetry.
  // OV_LPT (bitmask, default 0 = today's behaviour byte-for-byte):
  //   bit0  reverse the SYRK downdate issue order (largest block first)
  //   bit1  UpdaterMSCKF per-feature loop ordering (see UpdaterMSCKF.cpp)
  static const int g_lpt = [] { const char *e = std::getenv("OV_LPT"); return e ? atoi(e) : 0; }();
  // OV_UPD_VERIFY=1: recompute the thread-count-DEPENDENT sites at nthr=4 and memcmp, so the
  // "exact math" claim is proven IN-BINARY rather than argued.
  static const bool g_updver = [] { const char *e = std::getenv("OV_UPD_VERIFY"); return e && *e == '1'; }();
  static const int chold = [] { const char *e = std::getenv("OV_CHOL_DOWN"); return e ? atoi(e) : 0; }();
  if (chold > 0) {
    Eigen::LLT<Eigen::MatrixXd, Eigen::Upper> llt;
    { VPROF("7.ekf/d2_chol");
    llt.compute(S.selfadjointView<Eigen::Upper>()); }
    if (llt.info() == Eigen::Success) {
      const int nthr = chold > 1 ? chold : 4;
      const int Nst = (int)state->_Cov.rows(), mm = (int)res.rows();
      Eigen::MatrixXd Ma_pre_; // OV_UPD_VERIFY only
      if (g_updver) Ma_pre_ = M_a;
      { VPROF("7.ekf/e2_Vsolve");
      const int rb = (Nst + nthr - 1) / nthr;
#pragma omp parallel for schedule(static) num_threads(nthr)
      for (int t = 0; t < nthr; t++) {
        const int r0 = t * rb, nr = std::min(rb, Nst - r0);
        if (nr > 0) {
          auto blk = M_a.middleRows(r0, nr);
          llt.matrixU().template solveInPlace<Eigen::OnTheRight>(blk);
        }
      } }
      // M_a now holds V
      if (g_updver) {
        // rb = ceil(Nst/nthr) sets the row-block handed to the triangular solve, so the block
        // SHAPES change with the thread count. Redo the identical solve at the reference nthr=4
        // and compare byte-for-byte.
        const int rthr = 4, rb4 = (Nst + rthr - 1) / rthr;
#pragma omp parallel for schedule(static) num_threads(rthr)
        for (int t = 0; t < rthr; t++) {
          const int r0 = t * rb4, nr = std::min(rb4, Nst - r0);
          if (nr > 0) {
            auto blk = Ma_pre_.middleRows(r0, nr);
            llt.matrixU().template solveInPlace<Eigen::OnTheRight>(blk);
          }
        }
        double dmax = (Ma_pre_ - Eigen::MatrixXd(M_a)).cwiseAbs().maxCoeff();
        bool bx = (std::memcmp(Ma_pre_.data(), M_a.data(), sizeof(double) * (size_t)Nst * mm) == 0);
        printf("[updverify] Vsolve N=%d m=%d nthr=%d dmax=%.3e %s\n", Nst, mm, nthr, dmax,
               bx ? "BITEXACT" : "DIFFER");
      }
      { VPROF("7.ekf/f2_syrk_down");
      bool done_gpu = false;
      if (ekf_ds_available() && (long)Nst * Nst * mm / 2 > ds_min * 1000000L) {
        VPROF("7.ekf/f2_syrk_dsgpu");
        // contiguous Nst x Nst target carved out of a grown scratch (kernel assumes ldc == M);
        // only the UPPER triangle of D is valid after a sym call -- read nothing else.
        static thread_local std::vector<double> ds_scratch;
        if (ds_scratch.size() < (size_t)Nst * Nst) ds_scratch.resize((size_t)Nst * Nst);
        if (ekf_ds_gemm_nt(M_a.data(), Nst, mm, M_a.data(), Nst, ds_scratch.data(), true)) {
          Eigen::Map<Eigen::MatrixXd> D(ds_scratch.data(), Nst, Nst);
          state->_Cov.triangularView<Eigen::Upper>() -= D;
          done_gpu = true;
        }
      }
      // compensated cuBLAS syrk (fp32 hi/lo split, fp64 device sum): ~1e-6 relative on the
      // downdate, forward-stable (no S^-1 amplification). Fleet-gate before shipping.
      static const long gpusyrk_min = [] { const char *e = std::getenv("OV_GPU_SYRK_MIN_MMAC"); return e ? atol(e) : 30L; }();
      if (!done_gpu && ekf_gpu_syrk_enabled() && (long)Nst * Nst * mm / 2 > gpusyrk_min * 1000000L) {
        VPROF("7.ekf/f2_syrk_gpucomp");
        static thread_local std::vector<double> gs_scratch;
        if (gs_scratch.size() < (size_t)Nst * Nst) gs_scratch.resize((size_t)Nst * Nst);
        if (ekf_gpu_syrk(M_a.data(), Nst, mm, gs_scratch.data())) {
          Eigen::Map<Eigen::MatrixXd> D(gs_scratch.data(), Nst, Nst);
          state->_Cov.triangularView<Eigen::Upper>() -= D;
          done_gpu = true;
        }
      }
      if (!done_gpu) {
      const int cb = 96;
      const int nblk = (Nst + cb - 1) / cb;
      double *cov = state->_Cov.data();
      const int ld = (int)state->_Cov.outerStride();
      // OV_LPT&1 (LPT / longest-processing-time-first): work in block b is proportional to ce*w
      // with ce = c0 + w, i.e. it grows MONOTONICALLY with b, and schedule(dynamic) issues
      // iterations in index order -- so the default hands out the SMALLEST block first and the
      // LARGEST last, the worst possible order for makespan. Reversing the issue order is
      // BIT-EXACT: cb is fixed at 96 so the blocks, the GEMM shapes and the (disjoint) output
      // regions are all identical; only which thread runs which block, and when, changes.
      const bool lpt_dd = (g_lpt & 1) != 0;
      // OV_LPT_VERIFY=1: run the SAME blocks in the OPPOSITE order into an independent copy of
      // the covariance and memcmp the whole matrix. This is the in-binary proof that the LPT
      // reordering is bit-exact, not an argument that it ought to be.
      static const bool g_lptver = [] { const char *e = std::getenv("OV_LPT_VERIFY"); return e && *e == '1'; }();
      Eigen::MatrixXd Cref_;
      if (g_lptver) {
        Cref_ = state->_Cov;
        double *cref = Cref_.data();
        const int ld2 = (int)Cref_.outerStride();
#pragma omp parallel for schedule(dynamic) num_threads(nthr)
        for (int bb = 0; bb < nblk; bb++) {
          const int b = lpt_dd ? bb : (nblk - 1 - bb); // the OTHER order
          const int c0 = b * cb, w = std::min(cb, Nst - c0), ce = c0 + w;
          Eigen::Map<Eigen::MatrixXd, 0, Eigen::OuterStride<>> blkP(cref + (size_t)c0 * ld2, ce, w, Eigen::OuterStride<>(ld2));
          blkP.noalias() -= M_a.topRows(ce) * M_a.middleRows(c0, w).transpose();
        }
      }
#pragma omp parallel for schedule(dynamic) num_threads(nthr)
      for (int bb = 0; bb < nblk; bb++) {
        const int b = lpt_dd ? (nblk - 1 - bb) : bb;
        const int c0 = b * cb, w = std::min(cb, Nst - c0), ce = c0 + w;
        Eigen::Map<Eigen::MatrixXd, 0, Eigen::OuterStride<>> blkP(cov + (size_t)c0 * ld, ce, w, Eigen::OuterStride<>(ld));
        blkP.noalias() -= M_a.topRows(ce) * M_a.middleRows(c0, w).transpose();
      }
      if (g_lptver) {
        double dmax = (Cref_ - state->_Cov).cwiseAbs().maxCoeff();
        bool bx = (Cref_.outerStride() == state->_Cov.outerStride()) &&
                  (std::memcmp(Cref_.data(), state->_Cov.data(),
                               sizeof(double) * (size_t)Cref_.outerStride() * (size_t)Cref_.cols()) == 0);
        printf("[lptverify] downdate Nst=%d nblk=%d nthr=%d lpt=%d dmax=%.3e %s\n", Nst, nblk, nthr,
               (int)lpt_dd, dmax, bx ? "BITEXACT" : "DIFFER");
      }
      }
      if (scr) symmetrize_upper_to_lower(state->_Cov.data(), Nst, (int)state->_Cov.outerStride());
      else state->_Cov = state->_Cov.selfadjointView<Eigen::Upper>(); }

      Eigen::VectorXd diags2 = state->_Cov.diagonal();
      for (int i = 0; i < diags2.rows(); i++) {
        if (diags2(i) < 0.0) {
          PRINT_WARNING(RED "StateHelper::EKFUpdate() [chol] - diagonal at %d is %.2f\n" RESET, i, diags2(i));
          std::exit(EXIT_FAILURE);
        }
      }
      Eigen::VectorXd u = res;
      llt.matrixL().solveInPlace(u);
      Eigen::VectorXd dx2 = M_a * u;
      if (dx_out) *dx_out = dx2;
      for (size_t i = 0; i < state->_variables.size(); i++)
        state->_variables.at(i)->update(dx2.block(state->_variables.at(i)->id(), 0, state->_variables.at(i)->size(), 1));
      if (state->_options.do_calib_camera_intrinsics)
        for (auto const &calib : state->_cam_intrinsics)
          state->_cam_intrinsics_cameras.at(calib.first)->set_value(calib.second->value());
      state->push_nonradial_to_cameras();
      (void)mm;
      return;
    }
    PRINT_WARNING(YELLOW "StateHelper::EKFUpdate() [chol] - LLT failed, falling back\n" RESET);
  }

  // Invert our S (should we use a more stable method here??)
  Eigen::MatrixXd K;
  static const bool ksolve = [] { const char *e = std::getenv("OV_K_SOLVE"); return e && *e == '1'; }();
  if (ksolve) {
    // K^T = S^{-1} M^T via one Cholesky solve: skips the explicit inverse AND the extra GEMM.
    VPROF("7.ekf/e_K=M*Sinv");
    K = S.selfadjointView<Eigen::Upper>().llt().solve(M_a.transpose()).transpose();
  } else {
    Eigen::MatrixXd Sinv = Eigen::MatrixXd::Identity(m_rows_, m_rows_);
    { VPROF("7.ekf/d_Sinv_llt");
    S.selfadjointView<Eigen::Upper>().llt().solveInPlace(Sinv); }
    { VPROF("7.ekf/e_K=M*Sinv");
    K = M_a * Sinv.selfadjointView<Eigen::Upper>(); }
  }
  // Eigen::MatrixXd K = M_a * S.inverse();

  // Update Covariance
  { VPROF("7.ekf/f_cov_update");
  // Full GEMM (OMP-threaded by Eigen) then subtract: 2x flops of the triangular product but
  // 4x threads -> net faster, and bitwise math per element is unchanged.
  static const bool fulld = [] { const char *e = std::getenv("OV_FULL_DOWNDATE"); return e && *e == '1'; }();
  if (fulld) {
    Eigen::MatrixXd D_;
    D_.noalias() = K * M_a.transpose();
    state->_Cov.triangularView<Eigen::Upper>() -= D_;
    if (scr) symmetrize_upper_to_lower(state->_Cov.data(), N_st_, (int)state->_Cov.outerStride());
    else state->_Cov = state->_Cov.selfadjointView<Eigen::Upper>();
  } else {
    state->_Cov.triangularView<Eigen::Upper>() -= K * M_a.transpose();
    if (scr) symmetrize_upper_to_lower(state->_Cov.data(), N_st_, (int)state->_Cov.outerStride());
    else state->_Cov = state->_Cov.selfadjointView<Eigen::Upper>();
  } }
  // Cov -= K * M_a.transpose();
  // Cov = 0.5*(Cov+Cov.transpose());

  // We should check if we are not positive semi-definitate (i.e. negative diagionals is not s.p.d)
  Eigen::VectorXd diags = state->_Cov.diagonal();
  bool found_neg = false;
  for (int i = 0; i < diags.rows(); i++) {
    if (diags(i) < 0.0) {
      PRINT_WARNING(RED "StateHelper::EKFUpdate() - diagonal at %d is %.2f\n" RESET, i, diags(i));
      found_neg = true;
    }
  }
  if (found_neg) {
    std::exit(EXIT_FAILURE);
  }

  // Calculate our delta and update all our active states
  Eigen::VectorXd dx = K * res;
  if (dx_out) *dx_out = dx;
  for (size_t i = 0; i < state->_variables.size(); i++) {
    state->_variables.at(i)->update(dx.block(state->_variables.at(i)->id(), 0, state->_variables.at(i)->size(), 1));
  }

  // If we are doing online intrinsic calibration we should update our camera objects
  // NOTE: is this the best place to put this update logic??? probably..
  if (state->_options.do_calib_camera_intrinsics) {
    for (auto const &calib : state->_cam_intrinsics) {
      state->_cam_intrinsics_cameras.at(calib.first)->set_value(calib.second->value());
    }
  }
  state->push_nonradial_to_cameras();
}

void StateHelper::EKFUpdate(std::shared_ptr<State> state, const std::vector<std::shared_ptr<Type>> &H_order, const Eigen::MatrixXd &H,
                            const Eigen::VectorXd &res, const Eigen::MatrixXd &R) {
  ekf_update_impl(state, H_order, H, res, &R, 0.0);
}

void StateHelper::EKFUpdate(std::shared_ptr<State> state, const std::vector<std::shared_ptr<Type>> &H_order, const Eigen::MatrixXd &H,
                            const Eigen::VectorXd &res, double sigma2, Eigen::VectorXd *dx_out) {
  ekf_update_impl(state, H_order, H, res, nullptr, sigma2, dx_out);
}

// OV_CHUNK_ROWS: sequential chunked EKF update. EXACT for isotropic R, no relinearisation.
//
// measurement_compress_inplace applies an orthogonal Q^T to [H|res], and Q^T (sigma^2 I) Q =
// sigma^2 I, so the compressed system still has isotropic noise. Any ROW partition of an
// isotropic-R system is block-diagonal in R, hence the blocks are conditionally independent:
// p(x|z1,z2) ~ p(z2|x) p(z1|x) p(x). Updating with block 1 and using that posterior as the prior
// for block 2 therefore reproduces the batch posterior exactly, and EKFUpdate takes H and res as
// fixed inputs so nothing is re-evaluated between sub-updates.
//
// The win comes from the SHAPE, not the algebra: after compression H is square and upper
// triangular, so rows [r0, r0+b) are exactly zero in columns [0, r0) and the sub-update only has
// to touch the column SUFFIX. Columns are laid out in H_order order, so a column suffix is a
// contiguous H_order suffix; we snap c0 down to a variable boundary (c0 <= r0 always, so the
// few included columns below r0 are exact stored zeros -- conservative, never wrong).
void StateHelper::EKFUpdateTriChunked(std::shared_ptr<State> state, const std::vector<std::shared_ptr<Type>> &H_order,
                                      const Eigen::MatrixXd &H, const Eigen::VectorXd &res, double sigma_pix_sq, int chunk_rows) {
  const int m = (int)H.rows(), c = (int)H.cols();
  // Not compressed (H not square) or too small to be worth splitting -> plain batch update.
  if (m != c || chunk_rows <= 0 || m <= chunk_rows) {
    StateHelper::EKFUpdate(state, H_order, H, res, sigma_pix_sq);
    return;
  }
  std::vector<int> off(H_order.size());
  int t = 0;
  for (size_t i = 0; i < H_order.size(); i++) { off[i] = t; t += (int)H_order[i]->size(); }

  auto run_chunked = [&]() {
    // RESIDUAL RE-EVALUATION -- this is what makes the split exact, and it is easy to miss.
    // Sequential conditioning gives the batch posterior only if chunk j's innovation is measured
    // against the state chunk j-1 already moved:  z_j - H_j*x_{j-1} = res_j - H_j*(sum dx_{<j}).
    // The COVARIANCE is exact without it (it has no residual dependence), so a covariance-only
    // check looks perfect while the mean is wrong -- measured dX ~5e-2 before this term was added.
    // H itself is never re-evaluated, so this is not relinearisation: it is the same linear
    // system, just conditioned in two steps.
    Eigen::VectorXd dx_tot = Eigen::VectorXd::Zero(state->_Cov.rows());
    Eigen::VectorXd dx_i, dxH;
    size_t v0 = 0; // monotone across chunks -- never reset, the suffix only shrinks
    for (int r0 = 0; r0 < m;) {
      const int b = std::min(chunk_rows, m - r0);
      while (v0 + 1 < H_order.size() && off[v0 + 1] <= r0) v0++;
      const int c0 = off[v0], w = c - c0;
      std::vector<std::shared_ptr<Type>> sub(H_order.begin() + v0, H_order.end());
      Eigen::MatrixXd Hs = H.block(r0, c0, b, w);
      Eigen::VectorXd rs = res.segment(r0, b);
      if (r0 > 0) {
        // gather the accumulated correction into this chunk's H column layout
        dxH.resize(w);
        for (size_t i = v0; i < H_order.size(); i++)
          dxH.segment(off[i] - c0, H_order[i]->size()) = dx_tot.segment(H_order[i]->id(), H_order[i]->size());
        rs.noalias() -= Hs * dxH;
      }
      // Each sub-update re-gathers P (correct and required: P has changed) and re-symmetrises.
      StateHelper::EKFUpdate(state, sub, Hs, rs, sigma_pix_sq, &dx_i);
      dx_tot += dx_i;
      r0 += b;
    }
  };

  // OV_CHUNK_VERIFY: run the SAME measurement both ways from the SAME prior and report the
  // deviation. Snapshot/restore is required because the pipeline is GPU-nondeterministic
  // run-to-run (k_select atomicAdd), so a cross-run trajectory diff proves nothing.
  static const bool chunk_verify = [] { const char *e = std::getenv("OV_CHUNK_VERIFY"); return e && *e == '1'; }();
  if (chunk_verify) {
    Eigen::MatrixXd Cov0 = state->_Cov;
    std::vector<Eigen::MatrixXd> val0;
    val0.reserve(state->_variables.size());
    for (auto &v : state->_variables) val0.push_back(v->value());

    Eigen::VectorXd dx_ref;
    StateHelper::EKFUpdate(state, H_order, H, res, sigma_pix_sq, &dx_ref); // batch reference
    Eigen::MatrixXd Cov_b = state->_Cov;
    std::vector<Eigen::MatrixXd> val_b;
    val_b.reserve(state->_variables.size());
    for (auto &v : state->_variables) val_b.push_back(v->value());

    state->_Cov = Cov0; // restore the prior
    for (size_t i = 0; i < state->_variables.size(); i++) state->_variables[i]->set_value(val0[i]);

    run_chunked();

    const double cov_scale = Cov_b.cwiseAbs().maxCoeff();
    const double dcov = (state->_Cov - Cov_b).cwiseAbs().maxCoeff();
    double dx = 0.0, xscale = 0.0;
    for (size_t i = 0; i < state->_variables.size(); i++) {
      dx = std::max(dx, (state->_variables[i]->value() - val_b[i]).cwiseAbs().maxCoeff());
      xscale = std::max(xscale, val_b[i].cwiseAbs().maxCoeff());
    }
    // dxmax is the size of the batch correction itself. If the residue is the manifold
    // composition term (k small quaternion corrections instead of one), it must be SECOND order:
    // dX ~ dxmax^2. If it were an algebra error it would be first order, dX ~ dxmax.
    const double dxmax = dx_ref.cwiseAbs().maxCoeff();
    printf("[chunkverify] N=%d m=%d k=%d dCov=%.3e (rel %.3e) dX=%.3e (rel %.3e) dxmax=%.3e ratio=%.3e\n",
           (int)state->_Cov.rows(), m, (m + chunk_rows - 1) / chunk_rows, dcov,
           cov_scale > 0 ? dcov / cov_scale : 0.0, dx, xscale > 0 ? dx / xscale : 0.0,
           dxmax, dxmax > 0 ? dx / (dxmax * dxmax) : 0.0);
    fflush(stdout);
    return;
  }

  run_chunked();
}

void StateHelper::set_initial_covariance(std::shared_ptr<State> state, const Eigen::MatrixXd &covariance,
                                         const std::vector<std::shared_ptr<ov_type::Type>> &order) {
  ov_msckf::g_state_epoch.fetch_add(1, std::memory_order_relaxed); // OV_PREJAC cache invalidation

  // We need to loop through each element and overwrite the current covariance values
  // For example consider the following:
  // x = [ ori pos ] -> insert into -> x = [ ori bias pos ]
  // P = [ P_oo P_op ] -> P = [ P_oo  0   P_op ]
  //     [ P_po P_pp ]        [  0    P*    0  ]
  //                          [ P_po  0   P_pp ]
  // The key assumption here is that the covariance is block diagonal (cross-terms zero with P* can be dense)
  // This is normally the care on startup (for example between calibration and the initial state

  // For each variable, lets copy over all other variable cross terms
  // Note: this copies over itself to when i_index=k_index
  int i_index = 0;
  for (size_t i = 0; i < order.size(); i++) {
    int k_index = 0;
    for (size_t k = 0; k < order.size(); k++) {
      state->_Cov.block(order[i]->id(), order[k]->id(), order[i]->size(), order[k]->size()) =
          covariance.block(i_index, k_index, order[i]->size(), order[k]->size());
      k_index += order[k]->size();
    }
    i_index += order[i]->size();
  }
  state->_Cov = state->_Cov.selfadjointView<Eigen::Upper>();
}

void StateHelper::get_marginal_covariance_into(std::shared_ptr<State> state,
                                               const std::vector<std::shared_ptr<Type>> &small_variables,
                                               Eigen::Ref<Eigen::MatrixXd> out) {

  // Identical double loop to get_marginal_covariance() below. The loop writes EVERY (i,k)
  // block of the cov_size x cov_size result, so the Zero() there is pure waste (c^2 elements
  // per feature); `out` is caller-owned scratch and is left uninitialised on entry.
  int i_index = 0;
  for (size_t i = 0; i < small_variables.size(); i++) {
    int k_index = 0;
    for (size_t k = 0; k < small_variables.size(); k++) {
      out.block(i_index, k_index, small_variables[i]->size(), small_variables[k]->size()) =
          state->_Cov.block(small_variables[i]->id(), small_variables[k]->id(), small_variables[i]->size(), small_variables[k]->size());
      k_index += small_variables[k]->size();
    }
    i_index += small_variables[i]->size();
  }
}

Eigen::MatrixXd StateHelper::get_marginal_covariance(std::shared_ptr<State> state,
                                                     const std::vector<std::shared_ptr<Type>> &small_variables) {

  // Calculate the marginal covariance size we need to make our matrix
  int cov_size = 0;
  for (size_t i = 0; i < small_variables.size(); i++) {
    cov_size += small_variables[i]->size();
  }

  // Construct our return covariance
  Eigen::MatrixXd Small_cov = Eigen::MatrixXd::Zero(cov_size, cov_size);

  // For each variable, lets copy over all other variable cross terms
  // Note: this copies over itself to when i_index=k_index
  int i_index = 0;
  for (size_t i = 0; i < small_variables.size(); i++) {
    int k_index = 0;
    for (size_t k = 0; k < small_variables.size(); k++) {
      Small_cov.block(i_index, k_index, small_variables[i]->size(), small_variables[k]->size()) =
          state->_Cov.block(small_variables[i]->id(), small_variables[k]->id(), small_variables[i]->size(), small_variables[k]->size());
      k_index += small_variables[k]->size();
    }
    i_index += small_variables[i]->size();
  }

  // Return the covariance
  // Small_cov = 0.5*(Small_cov+Small_cov.transpose());
  return Small_cov;
}

Eigen::MatrixXd StateHelper::get_full_covariance(std::shared_ptr<State> state) {

  // Size of the covariance is the active
  int cov_size = (int)state->_Cov.rows();

  // Construct our return covariance
  Eigen::MatrixXd full_cov = Eigen::MatrixXd::Zero(cov_size, cov_size);

  // Copy in the active state elements
  full_cov.block(0, 0, state->_Cov.rows(), state->_Cov.rows()) = state->_Cov;

  // Return the covariance
  return full_cov;
}

void StateHelper::marginalize(std::shared_ptr<State> state, std::shared_ptr<Type> marg) {
  ov_msckf::g_state_epoch.fetch_add(1, std::memory_order_relaxed); // OV_PREJAC cache invalidation

  // Check if the current state has the element we want to marginalize
  if (std::find(state->_variables.begin(), state->_variables.end(), marg) == state->_variables.end()) {
    PRINT_ERROR(RED "StateHelper::marginalize() - Called on variable that is not in the state\n" RESET);
    PRINT_ERROR(RED "StateHelper::marginalize() - Marginalization, does NOT work on sub-variables yet...\n" RESET);
    std::exit(EXIT_FAILURE);
  }

  // Generic covariance has this form for x_1, x_m, x_2. If we want to remove x_m:
  //
  //  P_(x_1,x_1) P(x_1,x_m) P(x_1,x_2)
  //  P_(x_m,x_1) P(x_m,x_m) P(x_m,x_2)
  //  P_(x_2,x_1) P(x_2,x_m) P(x_2,x_2)
  //
  //  to
  //
  //  P_(x_1,x_1) P(x_1,x_2)
  //  P_(x_2,x_1) P(x_2,x_2)
  //
  // i.e. x_1 goes from 0 to marg_id, x_2 goes from marg_id+marg_size to Cov.rows() in the original covariance

  int marg_size = marg->size();
  int marg_id = marg->id();
  int x2_size = (int)state->_Cov.rows() - marg_id - marg_size;

  Eigen::MatrixXd Cov_new(state->_Cov.rows() - marg_size, state->_Cov.rows() - marg_size);

  // P_(x_1,x_1)
  Cov_new.block(0, 0, marg_id, marg_id) = state->_Cov.block(0, 0, marg_id, marg_id);

  // P_(x_1,x_2)
  Cov_new.block(0, marg_id, marg_id, x2_size) = state->_Cov.block(0, marg_id + marg_size, marg_id, x2_size);

  // P_(x_2,x_1)
  Cov_new.block(marg_id, 0, x2_size, marg_id) = Cov_new.block(0, marg_id, marg_id, x2_size).transpose();

  // P(x_2,x_2)
  Cov_new.block(marg_id, marg_id, x2_size, x2_size) = state->_Cov.block(marg_id + marg_size, marg_id + marg_size, x2_size, x2_size);

  // Now set new covariance
  // state->_Cov.resize(Cov_new.rows(),Cov_new.cols());
  state->_Cov = Cov_new;
  // state->Cov() = 0.5*(Cov_new+Cov_new.transpose());
  assert(state->_Cov.rows() == Cov_new.rows());

  // Now we keep the remaining variables and update their ordering
  // Note: DOES NOT SUPPORT MARGINALIZING SUBVARIABLES YET!!!!!!!
  std::vector<std::shared_ptr<Type>> remaining_variables;
  for (size_t i = 0; i < state->_variables.size(); i++) {
    // Only keep non-marginal states
    if (state->_variables.at(i) != marg) {
      if (state->_variables.at(i)->id() > marg_id) {
        // If the variable is "beyond" the marginal one in ordering, need to "move it forward"
        state->_variables.at(i)->set_local_id(state->_variables.at(i)->id() - marg_size);
      }
      remaining_variables.push_back(state->_variables.at(i));
    }
  }

  // Delete the old state variable to free up its memory
  // NOTE: we don't need to do this any more since our variable is a shared ptr
  // NOTE: thus this is automatically managed, but this allows outside references to keep the old variable
  // delete marg;
  marg->set_local_id(-1);

  // Now set variables as the remaining ones
  state->_variables = remaining_variables;
}

std::shared_ptr<Type> StateHelper::clone(std::shared_ptr<State> state, std::shared_ptr<Type> variable_to_clone) {
  ov_msckf::g_state_epoch.fetch_add(1, std::memory_order_relaxed); // OV_PREJAC cache invalidation

  // Get total size of new cloned variables, and the old covariance size
  int total_size = variable_to_clone->size();
  int old_size = (int)state->_Cov.rows();
  int new_loc = (int)state->_Cov.rows();

  // Resize both our covariance to the new size
  { // grow without zero-filling the old block twice (conservativeResizeLike zeroes everything)
    const int ns = old_size + total_size;
    Eigen::MatrixXd grown(ns, ns);
    grown.topLeftCorner(old_size, old_size) = state->_Cov;
    grown.bottomRows(total_size).setZero();
    grown.topRightCorner(old_size, total_size).setZero();
    state->_Cov = std::move(grown);
  }

  // What is the new state, and variable we inserted
  const std::vector<std::shared_ptr<Type>> new_variables = state->_variables;
  std::shared_ptr<Type> new_clone = nullptr;

  // Loop through all variables, and find the variable that we are going to clone
  for (size_t k = 0; k < state->_variables.size(); k++) {

    // Skip this if it is not the same
    // First check if the top level variable is the same, then check the sub-variables
    std::shared_ptr<Type> type_check = state->_variables.at(k)->check_if_subvariable(variable_to_clone);
    if (state->_variables.at(k) == variable_to_clone) {
      type_check = state->_variables.at(k);
    } else if (type_check != variable_to_clone) {
      continue;
    }

    // So we will clone this one
    int old_loc = type_check->id();

    // Copy the covariance elements
    state->_Cov.block(new_loc, new_loc, total_size, total_size) = state->_Cov.block(old_loc, old_loc, total_size, total_size);
    state->_Cov.block(0, new_loc, old_size, total_size) = state->_Cov.block(0, old_loc, old_size, total_size);
    state->_Cov.block(new_loc, 0, total_size, old_size) = state->_Cov.block(old_loc, 0, total_size, old_size);

    // Create clone from the type being cloned
    new_clone = type_check->clone();
    new_clone->set_local_id(new_loc);
    break;
  }

  // Check if the current state has this variable
  if (new_clone == nullptr) {
    PRINT_ERROR(RED "StateHelper::clone() - Called on variable is not in the state\n" RESET);
    PRINT_ERROR(RED "StateHelper::clone() - Ensure that the variable specified is a variable, or sub-variable..\n" RESET);
    std::exit(EXIT_FAILURE);
  }

  // Add to variable list and return
  state->_variables.push_back(new_clone);
  return new_clone;
}

namespace {
std::vector<std::vector<std::shared_ptr<ov_type::Type>>> g_initup_order;
std::vector<Eigen::MatrixXd> g_initup_H;
std::vector<Eigen::VectorXd> g_initup_r;
} // namespace

void StateHelper::flush_init_updates(std::shared_ptr<State> state, double sigma_pix_sq) {
  ov_msckf::g_state_epoch.fetch_add(1, std::memory_order_relaxed); // OV_PREJAC cache invalidation
  if (g_initup_H.empty()) return;
  VPROF("I.init/e_up_ekf_batched");
  // union variable order
  std::vector<std::shared_ptr<Type>> order;
  std::map<std::shared_ptr<Type>, size_t> pos;
  size_t cols = 0, rows = 0;
  for (auto &o : g_initup_order)
    for (auto &v : o)
      if (!pos.count(v)) { pos[v] = cols; order.push_back(v); cols += v->size(); }
  for (auto &H : g_initup_H) rows += H.rows();
  Eigen::MatrixXd Hb = Eigen::MatrixXd::Zero(rows, cols);
  Eigen::VectorXd rb(rows);
  size_t ro = 0;
  for (size_t i = 0; i < g_initup_H.size(); i++) {
    size_t co = 0;
    for (auto &v : g_initup_order[i]) {
      Hb.block(ro, pos[v], g_initup_H[i].rows(), v->size()) = g_initup_H[i].middleCols(co, v->size());
      co += v->size();
    }
    rb.segment(ro, g_initup_r[i].rows()) = g_initup_r[i];
    ro += g_initup_H[i].rows();
  }
  g_initup_order.clear(); g_initup_H.clear(); g_initup_r.clear();
  const long nrows_precmp = (long)Hb.rows();
  UpdaterHelper::measurement_compress_inplace(Hb, rb);
  if (Hb.rows() < 1) return;
  // Same OV_CHUNK_ROWS gate as UpdaterMSCKF: this path is another 5.5-8 ms of the whale frame
  // and goes through the identical compressed-triangular shape.
  static const int chunk_rows = [] { const char *e = std::getenv("OV_CHUNK_ROWS"); return e ? atoi(e) : 0; }();
  static const bool chunk_ok = [] { const char *e = std::getenv("OV_CHOL_DOWN"); return e && atoi(e) > 0; }();
  const bool compressed = (nrows_precmp > (long)Hb.cols());
  if (chunk_rows > 0 && chunk_ok && compressed && Hb.rows() == Hb.cols() && Hb.rows() >= 2 * chunk_rows) {
    StateHelper::EKFUpdateTriChunked(state, order, Hb, rb, sigma_pix_sq, chunk_rows);
  } else {
    StateHelper::EKFUpdate(state, order, Hb, rb, sigma_pix_sq);
  }
}

bool StateHelper::initialize(std::shared_ptr<State> state, std::shared_ptr<Type> new_variable,
                             const std::vector<std::shared_ptr<Type>> &H_order, Eigen::MatrixXd &H_R, Eigen::MatrixXd &H_L,
                             Eigen::MatrixXd &R, Eigen::VectorXd &res, double chi_2_mult) {
  ov_msckf::g_state_epoch.fetch_add(1, std::memory_order_relaxed); // OV_PREJAC cache invalidation

  // Check that this new variable is not already initialized
  if (std::find(state->_variables.begin(), state->_variables.end(), new_variable) != state->_variables.end()) {
    PRINT_ERROR("StateHelper::initialize_invertible() - Called on variable that is already in the state\n");
    PRINT_ERROR("StateHelper::initialize_invertible() - Found this variable at %d in covariance\n", new_variable->id());
    std::exit(EXIT_FAILURE);
  }

  // Check that we have isotropic noise (i.e. is diagonal and all the same value)
  // TODO: can we simplify this so it doesn't take as much time?
  assert(R.rows() == R.cols());
  assert(R.rows() > 0);
  for (int r = 0; r < R.rows(); r++) {
    for (int c = 0; c < R.cols(); c++) {
      if (r == c && R(0, 0) != R(r, c)) {
        PRINT_ERROR(RED "StateHelper::initialize() - Your noise is not isotropic!\n" RESET);
        PRINT_ERROR(RED "StateHelper::initialize() - Found a value of %.2f verses value of %.2f\n" RESET, R(r, c), R(0, 0));
        std::exit(EXIT_FAILURE);
      } else if (r != c && R(r, c) != 0.0) {
        PRINT_ERROR(RED "StateHelper::initialize() - Your noise is not diagonal!\n" RESET);
        PRINT_ERROR(RED "StateHelper::initialize() - Found a value of %.2f at row %d and column %d\n" RESET, R(r, c), r, c);
        std::exit(EXIT_FAILURE);
      }
    }
  }

  //==========================================================
  //==========================================================
  // First we perform QR givens to seperate the system
  // The top will be a system that depends on the new state, while the bottom does not
  size_t new_var_size = new_variable->size();
  assert((int)new_var_size == H_L.cols());

  VPROF("I.init/TOTAL");
  static const bool hh_init = [] { const char *e = std::getenv("OV_QR_HH"); return e && *e == '1'; }();
  if (hh_init) {
    // Householder separation: identical subspace split as the Givens sweep (rows of the
    // triangularised system may differ by sign, applied consistently to H_R and res).
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(H_L);
    Eigen::MatrixXd Qt_HR = qr.householderQ().transpose() * H_R;
    Eigen::VectorXd Qt_res = qr.householderQ().transpose() * res;
    H_L = qr.matrixQR().triangularView<Eigen::Upper>();
    H_R = Qt_HR;
    res = Qt_res;
  } else {
  Eigen::JacobiRotation<double> tempHo_GR;
  for (int n = 0; n < H_L.cols(); ++n) {
    for (int m = (int)H_L.rows() - 1; m > n; m--) {
      // Givens matrix G
      tempHo_GR.makeGivens(H_L(m - 1, n), H_L(m, n));
      (H_L.block(m - 1, n, 2, H_L.cols() - n)).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
      (res.block(m - 1, 0, 2, 1)).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
      (H_R.block(m - 1, 0, 2, H_R.cols())).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
    }
  }
  }

  // Separate into initializing and updating portions
  // 1. Invertible initializing system
  Eigen::MatrixXd Hxinit = H_R.block(0, 0, new_var_size, H_R.cols());
  Eigen::MatrixXd H_finit = H_L.block(0, 0, new_var_size, new_var_size);
  Eigen::VectorXd resinit = res.block(0, 0, new_var_size, 1);
  Eigen::MatrixXd Rinit = R.block(0, 0, new_var_size, new_var_size);

  // 2. Nullspace projected updating system
  Eigen::MatrixXd Hup = H_R.block(new_var_size, 0, H_R.rows() - new_var_size, H_R.cols());
  Eigen::VectorXd resup = res.block(new_var_size, 0, res.rows() - new_var_size, 1);
  Eigen::MatrixXd Rup = R.block(new_var_size, new_var_size, R.rows() - new_var_size, R.rows() - new_var_size);

  //==========================================================
  //==========================================================

  // Do mahalanobis distance testing
  VPROF("I.init/c_mahal_start");
  Eigen::MatrixXd P_up = get_marginal_covariance(state, H_order);
  assert(Rup.rows() == Hup.rows());
  assert(Hup.cols() == P_up.cols());
  Eigen::MatrixXd S = Hup * P_up * Hup.transpose() + Rup;
  double chi2 = resup.dot(S.llt().solve(resup));

  // Get what our threshold should be
  boost::math::chi_squared chi_squared_dist(res.rows());
  double chi2_check = boost::math::quantile(chi_squared_dist, 0.95);
  if (chi2 > chi_2_mult * chi2_check) {
    return false;
  }

  //==========================================================
  //==========================================================
  // Finally, initialize it in our state
  { VPROF("I.init/d_invertible");
  StateHelper::initialize_invertible(state, new_variable, H_order, Hxinit, H_finit, Rinit, resinit); }

  // Update with updating portion
  if (Hup.rows() > 0) {
    // Batch mode: accumulate every init's update-portion and apply ONE stacked EKFUpdate at
    // the end of the delayed-init pass. ~900 full-covariance downdates/60s become ~300.
    // Later features' Jacobians then linearize before these corrections (one-pass lag);
    // statistically equivalent, ATE-gated.
    static const bool batch_up = [] { const char *e = std::getenv("OV_BATCH_INIT_UP"); return e && *e == '1'; }();
    if (batch_up) {
      g_initup_order.push_back(H_order);
      g_initup_H.push_back(Hup);
      g_initup_r.push_back(resup);
    } else {
      VPROF("I.init/e_up_ekf");
      StateHelper::EKFUpdate(state, H_order, Hup, resup, Rup);
    }
  }
  return true;
}

void StateHelper::initialize_invertible(std::shared_ptr<State> state, std::shared_ptr<Type> new_variable,
                                        const std::vector<std::shared_ptr<Type>> &H_order, const Eigen::MatrixXd &H_R,
                                        const Eigen::MatrixXd &H_L, const Eigen::MatrixXd &R, const Eigen::VectorXd &res) {
  ov_msckf::g_state_epoch.fetch_add(1, std::memory_order_relaxed); // OV_PREJAC cache invalidation

  // Check that this new variable is not already initialized
  if (std::find(state->_variables.begin(), state->_variables.end(), new_variable) != state->_variables.end()) {
    PRINT_ERROR("StateHelper::initialize_invertible() - Called on variable that is already in the state\n");
    PRINT_ERROR("StateHelper::initialize_invertible() - Found this variable at %d in covariance\n", new_variable->id());
    std::exit(EXIT_FAILURE);
  }

  // Check that we have isotropic noise (i.e. is diagonal and all the same value)
  // TODO: can we simplify this so it doesn't take as much time?
  assert(R.rows() == R.cols());
  assert(R.rows() > 0);
  for (int r = 0; r < R.rows(); r++) {
    for (int c = 0; c < R.cols(); c++) {
      if (r == c && R(0, 0) != R(r, c)) {
        PRINT_ERROR(RED "StateHelper::initialize_invertible() - Your noise is not isotropic!\n" RESET);
        PRINT_ERROR(RED "StateHelper::initialize_invertible() - Found a value of %.2f verses value of %.2f\n" RESET, R(r, c), R(0, 0));
        std::exit(EXIT_FAILURE);
      } else if (r != c && R(r, c) != 0.0) {
        PRINT_ERROR(RED "StateHelper::initialize_invertible() - Your noise is not diagonal!\n" RESET);
        PRINT_ERROR(RED "StateHelper::initialize_invertible() - Found a value of %.2f at row %d and column %d\n" RESET, R(r, c), r, c);
        std::exit(EXIT_FAILURE);
      }
    }
  }

  //==========================================================
  //==========================================================
  // Part of the Kalman Gain K = (P*H^T)*S^{-1} = M*S^{-1}
  assert(res.rows() == R.rows());
  assert(H_L.rows() == res.rows());
  assert(H_L.rows() == H_R.rows());
  Eigen::MatrixXd M_a = Eigen::MatrixXd::Zero(state->_Cov.rows(), res.rows());

  // Get the location in small jacobian for each measuring variable
  int current_it = 0;
  std::vector<int> H_id;
  for (const auto &meas_var : H_order) {
    H_id.push_back(current_it);
    current_it += meas_var->size();
  }

  //==========================================================
  //==========================================================
  // For each active variable find its M = P*H^T
  {
    // Single gathered GEMM (same fix as EKFUpdate's M_a; the per-variable loop issued
    // hundreds of tiny products per SLAM-feature initialization).
    Eigen::MatrixXd P_g(state->_Cov.rows(), H_R.cols());
    int cit2 = 0;
    std::vector<int> hid2;
    for (const auto &mv : H_order) { hid2.push_back(cit2); cit2 += mv->size(); }
    for (size_t i2 = 0; i2 < H_order.size(); i2++)
      P_g.block(0, hid2[i2], P_g.rows(), H_order[i2]->size()) =
          state->_Cov.block(0, H_order[i2]->id(), state->_Cov.rows(), H_order[i2]->size());
    M_a.noalias() = P_g * H_R.transpose();
  }

  //==========================================================
  //==========================================================
  // Get covariance of this small jacobian
  Eigen::MatrixXd P_small = StateHelper::get_marginal_covariance(state, H_order);

  // M = H_R*Cov*H_R' + R
  Eigen::MatrixXd M(H_R.rows(), H_R.rows());
  M.triangularView<Eigen::Upper>() = H_R * P_small * H_R.transpose();
  M.triangularView<Eigen::Upper>() += R;

  // Covariance of the variable/landmark that will be initialized
  assert(H_L.rows() == H_L.cols());
  assert(H_L.rows() == new_variable->size());
  Eigen::MatrixXd H_Linv = H_L.inverse();
  Eigen::MatrixXd P_LL = H_Linv * M.selfadjointView<Eigen::Upper>() * H_Linv.transpose();

  // Augment the covariance matrix
  size_t oldSize = state->_Cov.rows();
  { const int ns = oldSize + new_variable->size(), k = new_variable->size();
    Eigen::MatrixXd grown(ns, ns);
    grown.topLeftCorner(oldSize, oldSize) = state->_Cov;
    grown.bottomRows(k).setZero();
    grown.topRightCorner(oldSize, k).setZero();
    state->_Cov = std::move(grown);
  }
  state->_Cov.block(0, oldSize, oldSize, new_variable->size()).noalias() = -M_a * H_Linv.transpose();
  state->_Cov.block(oldSize, 0, new_variable->size(), oldSize) = state->_Cov.block(0, oldSize, oldSize, new_variable->size()).transpose();
  state->_Cov.block(oldSize, oldSize, new_variable->size(), new_variable->size()) = P_LL;

  // Update the variable that will be initialized (invertible systems can only update the new variable).
  // However this update should be almost zero if we already used a conditional Gauss-Newton to solve for the initial estimate
  new_variable->update(H_Linv * res);

  // Now collect results, and add it to the state variables
  new_variable->set_local_id(oldSize);
  state->_variables.push_back(new_variable);

  // std::stringstream ss;
  // ss << new_variable->id() <<  " init dx = " << (H_Linv * res).transpose() << std::endl;
  // PRINT_DEBUG(ss.str().c_str());
}

void StateHelper::augment_clone(std::shared_ptr<State> state, Eigen::Matrix<double, 3, 1> last_w) {
  ov_msckf::g_state_epoch.fetch_add(1, std::memory_order_relaxed); // OV_PREJAC cache invalidation

  // We can't insert a clone that occured at the same timestamp!
  if (state->_clones_IMU.find(state->_timestamp) != state->_clones_IMU.end()) {
    PRINT_ERROR(RED "TRIED TO INSERT A CLONE AT THE SAME TIME AS AN EXISTING CLONE, EXITING!#!@#!@#\n" RESET);
    std::exit(EXIT_FAILURE);
  }

  // Call on our cloner and add it to our vector of types
  // NOTE: this will clone the clone pose to the END of the covariance...
  std::shared_ptr<Type> posetemp = StateHelper::clone(state, state->_imu->pose());

  // Cast to a JPL pose type, check if valid
  std::shared_ptr<PoseJPL> pose = std::dynamic_pointer_cast<PoseJPL>(posetemp);
  if (pose == nullptr) {
    PRINT_ERROR(RED "INVALID OBJECT RETURNED FROM STATEHELPER CLONE, EXITING!#!@#!@#\n" RESET);
    std::exit(EXIT_FAILURE);
  }

  // Append the new clone to our clone vector
  state->_clones_IMU[state->_timestamp] = pose;

  // If we are doing time calibration, then our clones are a function of the time offset
  // Logic is based on Mingyang Li and Anastasios I. Mourikis paper:
  // http://journals.sagepub.com/doi/pdf/10.1177/0278364913515286
  if (state->_options.do_calib_camera_timeoffset) {
    // Jacobian to augment by
    Eigen::Matrix<double, 6, 1> dnc_dt = Eigen::MatrixXd::Zero(6, 1);
    dnc_dt.block(0, 0, 3, 1) = last_w;
    dnc_dt.block(3, 0, 3, 1) = state->_imu->vel();
    // Augment covariance with time offset Jacobian
    // TODO: replace this with a call to the EKFPropagate function instead....
    state->_Cov.block(0, pose->id(), state->_Cov.rows(), 6) +=
        state->_Cov.block(0, state->_calib_dt_CAMtoIMU->id(), state->_Cov.rows(), 1) * dnc_dt.transpose();
    state->_Cov.block(pose->id(), 0, 6, state->_Cov.rows()) +=
        dnc_dt * state->_Cov.block(state->_calib_dt_CAMtoIMU->id(), 0, 1, state->_Cov.rows());
  }
}

namespace ov_msckf {
extern std::atomic<long> g_slam_init_ok, g_slam_init_fail, g_slam_marg;
}

void StateHelper::marginalize_old_clone(std::shared_ptr<State> state) {
  if ((int)state->_clones_IMU.size() > state->_options.max_clone_size) {
    double marginal_time = state->margtimestep();
    // Lock the mutex to avoid deleting any elements from _clones_IMU while accessing it from other threads
    std::lock_guard<std::mutex> lock(state->_mutex_state);
    assert(marginal_time != INFINITY);
    StateHelper::marginalize(state, state->_clones_IMU.at(marginal_time));
    // Note that the marginalizer should have already deleted the clone
    // Thus we just need to remove the pointer to it from our state
    state->_clones_IMU.erase(marginal_time);
  }
}

void StateHelper::marginalize_slam(std::shared_ptr<State> state) {
  // Remove SLAM features that have their marginalization flag set
  // We also check that we do not remove any aruoctag landmarks
  int ct_marginalized = 0;
  auto it0 = state->_features_SLAM.begin();
  while (it0 != state->_features_SLAM.end()) {
    if ((*it0).second->should_marg && (int)(*it0).first > 4 * state->_options.max_aruco_features) {
      StateHelper::marginalize(state, (*it0).second);
      it0 = state->_features_SLAM.erase(it0);
      ct_marginalized++;
      ov_msckf::g_slam_marg.fetch_add(1, std::memory_order_relaxed);
    } else {
      it0++;
    }
  }
}