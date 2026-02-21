//! \file cbdiso_3d.cpp
//! \brief Problem generator for a viscous circumbinary accretion disk in 3D
//!
//! The disk is initialized with:
//! - Radial power-law density profile: rho ~ r^-3/2
//! - Vertical Gaussian structure: rho ~ exp(-z^2/2H^2)
//! - Keplerian rotation in the orbital plane
//! - Binary cavity




#include <math.h>
#include <algorithm>
#include <iostream>
#include <cstddef>
#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <Kokkos_MathematicalFunctions.hpp>

#include "parameter_input.hpp"
#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "coordinates/cell_locations.hpp"

// ===========================================
// ======= Kokkos compatible functions =======
// ===========================================

KOKKOS_INLINE_FUNCTION
Real wrap_time(Real t, Real period) {
  return t - Kokkos::floor(t / period) * period;
}

KOKKOS_INLINE_FUNCTION
Real kokkos_atan2(Real y, Real x) {
#if defined(__CUDA_ARCH__)
  return atan2f(y, x);  
#else
  return std::atan2(y, x);
#endif
}

// global variable declarations
namespace {
  Real g_mach;
  Real g_alpha;
  Real g_eps;
  Real g_floor_density;
  Real g_floor_pressure;
  bool g_sources_enabled;
}


// =====================================================
// ===== Problem-specific classes: Binary and Disk =====
// =====================================================

class Binary {
public:
  Binary(Real semimajoraxis, 
         Real eccentricity, 
         Real mass_ratio,
         Real TotalMass, 
         Real SinkRadius, 
         Real SinkRate,
         Real SofteningRadius,
         int SinkModel)
    : _semimajoraxis(semimajoraxis),
      _eccentricity(eccentricity),
      _mass_ratio(mass_ratio),
      _TotalMass(TotalMass),
      _SinkRadius(SinkRadius),
      _SinkRate(SinkRate),
      _SofteningRadius(SofteningRadius),
      _SinkType(SinkModel),
      _period(Period()),
      _mean_motion(MeanMotion()) {}

  // public properties 
  KOKKOS_INLINE_FUNCTION Real TotalMass() const { return _TotalMass; }
  KOKKOS_INLINE_FUNCTION Real m1()        const { return _TotalMass               / (1 + _mass_ratio); }
  KOKKOS_INLINE_FUNCTION Real m2()        const { return _TotalMass * _mass_ratio / (1 + _mass_ratio); }
  KOKKOS_INLINE_FUNCTION Real rsink1()    const { return _SinkRadius; }
  KOKKOS_INLINE_FUNCTION Real rsink2()    const { return _SinkRadius*_mass_ratio; }
  KOKKOS_INLINE_FUNCTION Real rsoft1()    const { return _SofteningRadius; }
  KOKKOS_INLINE_FUNCTION Real rsoft2()    const { return _SofteningRadius * _mass_ratio; }
  KOKKOS_INLINE_FUNCTION int SinkModel()  const { return _SinkType; }
  KOKKOS_INLINE_FUNCTION Real SinkRate()  const { return _SinkRate; }

  // Test particle orbital frequency 
  KOKKOS_INLINE_FUNCTION
  Real Omega(Real x, Real y, Real z, Kokkos::Array<Kokkos::Array<Real, 3>, 2> binary_position) const {
    Real G          = 1.0; // Gravitational constant in code units
    Kokkos::Array<Real, 3> r1_vec = {x -  binary_position[0][0], y -  binary_position[0][1], z -  binary_position[0][2]};
    Kokkos::Array<Real, 3> r2_vec = {x -  binary_position[1][0], y -  binary_position[1][1], z -  binary_position[1][2]};
    Real r1         = Kokkos::sqrt(r1_vec[0]*r1_vec[0] + r1_vec[1]*r1_vec[1] + r1_vec[2]*r1_vec[2]);
    Real r2         = Kokkos::sqrt(r2_vec[0]*r2_vec[0] + r2_vec[1]*r2_vec[1] + r2_vec[2]*r2_vec[2]);
    Real omega1     = Kokkos::sqrt(G * m1() / (r1*r1*r1 + g_eps));
    Real omega2     = Kokkos::sqrt(G * m2() / (r2*r2*r2 + g_eps));
    return omega1 + omega2;
  }

  // Gravitational potential of binary
  KOKKOS_INLINE_FUNCTION
  Real Potential(Real x, Real y, Real z, Kokkos::Array<Kokkos::Array<Real, 3>, 2> binary_position) const {
    Real G          = 1.0;  // Gravitational constant in code units
    Kokkos::Array<Real, 3> r1_vec = {x - binary_position[0][0], y - binary_position[0][1], z - binary_position[0][2]};
    Kokkos::Array<Real, 3> r2_vec = {x - binary_position[1][0], y - binary_position[1][1], z - binary_position[1][2]};
    Real phi1       = -G * m1() / Kokkos::sqrt(r1_vec[0]*r1_vec[0] + r1_vec[1]*r1_vec[1] + r1_vec[2]*r1_vec[2] + rsoft1()*rsoft1() + g_eps);
    Real phi2       = -G * m2() / Kokkos::sqrt(r2_vec[0]*r2_vec[0] + r2_vec[1]*r2_vec[1] + r2_vec[2]*r2_vec[2] + rsoft2()*rsoft2() + g_eps);

    return phi1 + phi2;
  }


  // Keplerians equation for eccentric binary
  KOKKOS_INLINE_FUNCTION
  Real KeplerEquation(Real E, Real e, Real n, Real t){
    return E - e * Kokkos::sin(E) - n * t;
  }

  // Calculate eccentric anomaly using Newton-Raphson
  KOKKOS_INLINE_FUNCTION
  Real solve_newton_rapheson(Real e, Real n, Real t){
    int iter   = 0;
    Real E     = n * t;
    Real func  = KeplerEquation(E, e, n, t); // guess E = n * t 
    Real deriv = 1 - e * Kokkos::cos(E);
    while (Kokkos::abs(func) > 1e-15){
        E    -= func / deriv;
        iter += 1;
        if (iter > 10){
            return E;                       // best guess if not converged
        }
      }
    return E;
  }

  // Binary position as a function of time
  KOKKOS_INLINE_FUNCTION
  Kokkos::Array<Kokkos::Array<Real, 3>, 2> Position(Real t){
    Real time_periapse = wrap_time(t, _period);
    Real a             = _semimajoraxis;
    Real q             = _mass_ratio;
    Real e             = _eccentricity;
    Real E             = solve_newton_rapheson(e, _mean_motion, time_periapse);

    Kokkos::Array<Real, 3> PrimaryPos   = { a*q/(1+q)*(Kokkos::cos(E)-e),  a*q/(1+q)*Kokkos::sqrt(1-e*e)*Kokkos::sin(E), 0.0};
    Kokkos::Array<Real, 3> SecondaryPos = {-a  /(1+q)*(Kokkos::cos(E)-e), -a  /(1+q)*Kokkos::sqrt(1-e*e)*Kokkos::sin(E), 0.0};
    return {PrimaryPos, SecondaryPos};
  }

  // Binary velocity as a function of time
  KOKKOS_INLINE_FUNCTION
  Kokkos::Array<Kokkos::Array<Real, 3>, 2> Velocity(Real t){
    Real a             = _semimajoraxis;
    Real q             = _mass_ratio;
    Real e             = _eccentricity;
    Real time_periapse = wrap_time(t, _period);
    Real E             = solve_newton_rapheson(e, _mean_motion, time_periapse);
    Real Edot          = _mean_motion / (1 - e * Kokkos::cos(E));
    
    Kokkos::Array<Real, 3> PrimaryVel   = {-a*q/(1+q)*Edot*Kokkos::sin(E),  a*q/(1+q)*Kokkos::sqrt(1-e*e)*Edot*Kokkos::cos(E), 0.0};
    Kokkos::Array<Real, 3> SecondaryVel = { a  /(1+q)*Edot*Kokkos::sin(E), -a  /(1+q)*Kokkos::sqrt(1-e*e)*Edot*Kokkos::cos(E), 0.0};
    return {PrimaryVel, SecondaryVel};
  }
  
  // Option: Live integrator using rk4?

private:
  Real _semimajoraxis;
  Real _eccentricity;
  Real _mass_ratio;
  Real _TotalMass;
  Real _SinkRadius;
  Real _SinkRate;
  Real _SofteningRadius;
  Real _mean_motion;
  Real _period;
  int _SinkType; 

  KOKKOS_INLINE_FUNCTION Real MeanMotion() const { return Kokkos::sqrt(TotalMass() / (_semimajoraxis*_semimajoraxis*_semimajoraxis)); }
  KOKKOS_INLINE_FUNCTION Real Period()     const { return 2 * 3.1415926 * Kokkos::sqrt(Kokkos::pow(_semimajoraxis, 3) / TotalMass()); }
}; 




class Disk {
public:
  Disk(Real Mach, 
       Real alpha, 
       const Binary& binary)
    : _Mach(Mach),
      _alpha(alpha),
      _binary(binary) {}
  
  
    // hydro requirements
  KOKKOS_INLINE_FUNCTION
  Real SoundSpeedSquare(Real x, Real y, Real z,
                        Kokkos::Array<Kokkos::Array<Real, 3>, 2> binary_position) const {
    Real cs2 = - _binary.Potential(x, y, z, binary_position) / (_Mach * _Mach);
    return cs2;
  }

  KOKKOS_INLINE_FUNCTION
  Real Viscosity(Real x, Real y, Real z, Real t,
                 Kokkos::Array<Kokkos::Array<Real, 3>, 2> binary_position) const {
    Real cs    = Kokkos::sqrt(SoundSpeedSquare(x, y, z, binary_position));
    Real H     = cs / _binary.Omega(x, y, z, binary_position);  // Scale height H = cs / Omega
    return _alpha * cs * H;
  }


  // initialisation functions
  KOKKOS_INLINE_FUNCTION
  Real InitialDensity_XY(Real r, Real z) const {
    Real GM       = 1.0 * (_binary.TotalMass());      
    Real rho0     = 1.0;                           // Density normalization
    Real h2       = 1.0 / _Mach / _Mach;           // Aspect ratio H/R
    Real cs2      = h2 * GM / (r+g_eps);
    Real vertical = Kokkos::exp((1/h2) * (1/Kokkos::sqrt(1.0 + z*z/(r*r+g_eps)) - 1.0));  
    Real cavity   = (r > 1.2) ? 1.0 : 0.000001;    
    return rho0 * Kokkos::pow(r, -1.5) * vertical * cavity;
  }

  KOKKOS_INLINE_FUNCTION
  Kokkos::Array<Real, 3> InitialVelocity_XY(Real x, Real y, Real z) const {
    Real r_cyl = Kokkos::sqrt(x*x + y*y);
    Real phi   = kokkos_atan2(y, x);
    Real v_phi = Kokkos::sqrt(1.0 * _binary.TotalMass() / (r_cyl + g_eps));  // Keplerian velocity
    Real vx    = -v_phi * Kokkos::sin(phi);
    Real vy    =  v_phi * Kokkos::cos(phi);
    Real vz    = 0.0;   // No vertical motion
    return {vx, vy, vz};
  }

  KOKKOS_INLINE_FUNCTION
  Real InitialDensity(Real x, Real y, Real z, Real Orientation) const {
    // Rotate (y, z) by Orientation about x-axis
    Real y_rot = y * Kokkos::cos(Orientation) - z * Kokkos::sin(Orientation);
    Real z_rot = y * Kokkos::sin(Orientation) + z * Kokkos::cos(Orientation);
    Real r_cyl = Kokkos::sqrt(x*x + y_rot*y_rot);
    Real z_cyl = z_rot;
    return InitialDensity_XY(r_cyl, z_cyl);
  }

  KOKKOS_INLINE_FUNCTION
  Kokkos::Array<Real, 3> InitialVelocity(Real x, Real y, Real z, Real Orientation) const {
    Kokkos::Array<Real, 3> vel_xy = InitialVelocity_XY(x, y, z);
    return {vel_xy[0]*Kokkos::cos(Orientation), vel_xy[1], vel_xy[0]*Kokkos::sin(Orientation)};
  }

private:
  Real _Mach;
  Real _alpha;
  Binary _binary;
};



// ========================================================
// ============= forward declarations =============
// ========================================================

void viscous_source_term(Mesh *pm, const Real beta_dt, Kokkos::Array<Kokkos::Array<Real, 3>, 2> binary_position);
void binary_source_term(Mesh *pm, const Real beta_dt,  Kokkos::Array<Kokkos::Array<Real, 3>, 2> binary_position, Kokkos::Array<Kokkos::Array<Real, 3>, 2> binary_velocity);
void enforce_isothermal_pressure(Mesh *pm);
void enforce_floors(Mesh *pm);


// global binary definition for source terms
namespace {
  std::unique_ptr<Binary> g_binary;
  std::unique_ptr<Disk>   g_disk;
}



// ========================================================
// ================ Source Term Definitions ===============
// ========================================================

void circumbinary_source_term(Mesh *pm, const Real beta_dt_local) {
  Kokkos::Array<Kokkos::Array<Real, 3>, 2> binary_position = g_binary->Position(pm->time);
  Kokkos::Array<Kokkos::Array<Real, 3>, 2> binary_velocity = g_binary->Velocity(pm->time);
  viscous_source_term(pm, beta_dt_local, binary_position);
  binary_source_term(pm , beta_dt_local, binary_position, binary_velocity);
  enforce_isothermal_pressure(pm);
  enforce_floors(pm);
}



// ========================================================
// ===== Main problem generator for circumbinary disk =====
// ========================================================

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  // ========== READ PARAMETERS FROM <problem> BLOCK ==========
  // Binary parameters
  Real a_binary     = pin->GetOrAddReal("problem", "a_binary"     , 1.0);
  Real q_mass       = pin->GetOrAddReal("problem", "q_mass"       , 1.0);
  Real eccentricity = pin->GetOrAddReal("problem", "eccentricity" , 0.0);
  Real r_soft       = pin->GetOrAddReal("problem", "r_soft"       , 0.05);
  Real r_sink       = pin->GetOrAddReal("problem", "r_sink"       , 0.05);
  Real m_total      = pin->GetOrAddReal("problem", "m_total"      , 1.0);
  Real sink_rate    = pin->GetOrAddReal("problem", "sink_rate"    , 1.0);
  int SinkModel     = pin->GetOrAddInteger("problem", "sink_model", 2  );

  // Disk parameters
  g_mach            = pin->GetOrAddReal("problem", "Mach"          , 10.0);
  g_alpha           = pin->GetOrAddReal("problem", "alpha"         , 0.1 );
  Real Orientation  = pin->GetOrAddReal("problem", "orientation"   , 0.0 ); // 0 = prograde, pi = retrograde

  // numberical parameters
  g_eps             = pin->GetOrAddReal("problem", "eps"            , 1e-10); // small number to prevent division by zero in potential and omega calculations
  g_floor_density   = pin->GetOrAddReal("problem", "floor_density"  , 1e-10);
  g_floor_pressure  = pin->GetOrAddReal("problem", "floor_pressure" , 1e-12);

  // ===== Instantiate binary and disk class =====
  Binary binary     = Binary(a_binary, eccentricity, q_mass, m_total, r_sink, sink_rate, r_soft, SinkModel);
  g_binary          = std::make_unique<Binary>(binary);

  // ===== Create Disk object =====
  Disk disk         = Disk(g_mach, g_alpha, *g_binary);
  g_disk            = std::make_unique<Disk>(disk);
  const auto binary_position0 = g_binary->Position(0.0);

  // ===== Define source terms =====
  g_sources_enabled = true;
  user_srcs         = true;
  user_srcs_func    = &circumbinary_source_term;

  // Skip initialization if this is a restart; the runtime hooks above still run.
  if (restart) return;

  // ========== GET MESH AND PHYSICS OBJECTS ==========
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is     = indcs.is; int &ie = indcs.ie;
  int &js     = indcs.js; int &je = indcs.je;
  int &ks     = indcs.ks; int &ke = indcs.ke;
  
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &size          = pmbp->pmb->mb_size;

  // ========== INITIALIZE HYDRO VARIABLES ==========
  if (pmbp->phydro != nullptr) {
    auto &w0_ = pmbp->phydro->w0;  // Primitive variables (density, velocity, pressure)

    // Parallel loop over all mesh blocks and cells
    par_for("pgen_circumbinary", 
            DevExeSpace(),               // CPU or GPU execution space
            0, (pmbp->nmb_thispack-1),   // Loop 1: m = mesh block index (0 to N-1)
            ks, ke,                      // Loop 2: k = z index
            js, je,                      // Loop 3: j = y index
            is, ie,                      // Loop 4: i = x index
    
    KOKKOS_LAMBDA(int m, int k, int j, int i) { 
    // Lambda function for every (m,k,j,i) combination:

      Real &x1min = size.d_view(m).x1min;                   // xmin
      Real &x1max = size.d_view(m).x1max;                   // xmax
      int nx1     = indcs.nx1;                              // nx
      Real x1v    = CellCenterX(i-is, nx1, x1min, x1max);   // x coordinate

      Real &x2min = size.d_view(m).x2min;                   // ymin
      Real &x2max = size.d_view(m).x2max;                   // ymax
      int nx2     = indcs.nx2;                              // ny
      Real x2v    = CellCenterX(j-js, nx2, x2min, x2max);   // y coordinate

      Real &x3min = size.d_view(m).x3min;                   // zmin
      Real &x3max = size.d_view(m).x3max;                   // zmax
      int nx3     = indcs.nx3;                              // nz
      Real x3v    = CellCenterX(k-ks, nx3, x3min, x3max);   // z coordinate


      // ===== Compute density =====
      Real rho      = disk.InitialDensity(x1v, x2v, x3v, Orientation);

      // ===== Compute velocity =====
      Kokkos::Array<Real, 3> vel  = disk.InitialVelocity(x1v, x2v, x3v, Orientation);
      Real vx       = vel[0];
      Real vy       = vel[1];
      Real vz       = vel[2];
      
      // ===== Compute pressure =====
      Real cs2      = disk.SoundSpeedSquare(x1v, x2v, x3v, binary_position0);
      Real pres     = rho * cs2;

      // ===== Set primitive variables =====
      w0_(m, IDN, k, j, i) = rho;              // Density
      w0_(m, IVX, k, j, i) = vx;               // Velocity x-component
      w0_(m, IVY, k, j, i) = vy;               // Velocity y-component
      w0_(m, IVZ, k, j, i) = vz;               // Velocity z-component
      w0_(m, IPR, k, j, i) = pres;             // Pressure

    }); 

    // ===== Convert primitives to conserved variables =====
    pmbp->phydro->peos->PrimToCons(w0_, pmbp->phydro->u0, is, ie, js, je, ks, ke);

  }  // End if phydro != nullptr

  // ========== OPTIONAL: USER-DEFINED FUNCTIONS ==========
  // Uncomment these if you add corresponding user function implementations:
  
  // pgen_final_func = &CircumbinaryFinalAnalysis;
  // user_bcs_func = &CircumbinaryUserBCs;
  // user_hist_func = &CircumbinaryHistory;

}  // End CircumbinaryDisk




// =====================================================
// ============= User defined source terms =============
// =====================================================

// 1. Viscous source term for alpha viscosity in circumbinary disk
void viscous_source_term(Mesh *pm, const Real beta_dt, Kokkos::Array<Kokkos::Array<Real, 3>, 2> binary_position) {
  if (!g_sources_enabled || beta_dt <= 0.0) {
    return;
  }

  if (pm == nullptr || pm->pmb_pack == nullptr || g_binary == nullptr) {
    return;
  }

  MeshBlockPack *pmbp = pm->pmb_pack;
  if (pmbp->phydro == nullptr) {
    return;
  }

  const Disk disk(g_mach, g_alpha, *g_binary);

  auto &indcs              = pm->mb_indcs;
  auto &size               = pmbp->pmb->mb_size;
  auto &w                  = pmbp->phydro->w0;
  auto &u                  = pmbp->phydro->u0;
  const bool has_y         = (indcs.nx2 > 1);
  const bool has_z         = (indcs.nx3 > 1);
  const bool update_energy = pmbp->phydro->peos->eos_data.is_ideal;
  const Real t_now         = pm->time;
  const Real beta_dt_local = beta_dt;

  par_for("disk_viscous_source",
          DevExeSpace(),
          0, (pmbp->nmb_thispack - 1),
          indcs.ks, indcs.ke,
          indcs.js, indcs.je,
          indcs.is, indcs.ie,
          KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const Real dx1 = size.d_view(m).dx1;
    const Real dx2 = size.d_view(m).dx2;
    const Real dx3 = size.d_view(m).dx3;

    const int ip = (i == indcs.ie) ? i : (i + 1);
    const int im = (i == indcs.is) ? i : (i - 1);
    const int jp = (j == indcs.je) ? j : (j + 1);
    const int jm = (j == indcs.js) ? j : (j - 1);
    const int kp = (k == indcs.ke) ? k : (k + 1);
    const int km = (k == indcs.ks) ? k : (k - 1);

    const Real inv_dx1_sq = 1.0 / (dx1 * dx1);
    const Real inv_dx2_sq = has_y ? 1.0 / (dx2 * dx2) : 0.0;
    const Real inv_dx3_sq = has_z ? 1.0 / (dx3 * dx3) : 0.0;

    const Real rho = w(m, IDN, k, j, i);

    const Real vx = w(m, IVX, k, j, i);
    const Real vy = w(m, IVY, k, j, i);
    const Real vz = w(m, IVZ, k, j, i);

    const Real second_x_vx = (w(m, IVX, k, j, ip) - 2.0 * vx + w(m, IVX, k, j, im)) * inv_dx1_sq;
    const Real second_y_vx = has_y ? (w(m, IVX, k, jp, i) - 2.0 * vx + w(m, IVX, k, jm, i)) * inv_dx2_sq : 0.0;
    const Real second_z_vx = has_z ? (w(m, IVX, kp, j, i) - 2.0 * vx + w(m, IVX, km, j, i)) * inv_dx3_sq : 0.0;

    const Real second_x_vy = (w(m, IVY, k, j, ip) - 2.0 * vy + w(m, IVY, k, j, im)) * inv_dx1_sq;
    const Real second_y_vy = has_y ? (w(m, IVY, k, jp, i) - 2.0 * vy + w(m, IVY, k, jm, i)) * inv_dx2_sq : 0.0;
    const Real second_z_vy = has_z ? (w(m, IVY, kp, j, i) - 2.0 * vy + w(m, IVY, km, j, i)) * inv_dx3_sq : 0.0;

    const Real second_x_vz = (w(m, IVZ, k, j, ip) - 2.0 * vz + w(m, IVZ, k, j, im)) * inv_dx1_sq;
    const Real second_y_vz = has_y ? (w(m, IVZ, k, jp, i) - 2.0 * vz + w(m, IVZ, k, jm, i)) * inv_dx2_sq : 0.0;
    const Real second_z_vz = has_z ? (w(m, IVZ, kp, j, i) - 2.0 * vz + w(m, IVZ, km, j, i)) * inv_dx3_sq : 0.0;

    const Real lap_vx = second_x_vx + second_y_vx + second_z_vx;
    const Real lap_vy = second_x_vy + second_y_vy + second_z_vy;
    const Real lap_vz = second_x_vz + second_y_vz + second_z_vz;

    const Real grad_div_x = second_x_vx;
    const Real grad_div_y = has_y ? second_y_vy : 0.0;
    const Real grad_div_z = has_z ? second_z_vz : 0.0;

    const Real x1v = CellCenterX(i - indcs.is, indcs.nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    const Real x2v = CellCenterX(j - indcs.js, indcs.nx2, size.d_view(m).x2min, size.d_view(m).x2max);
    const Real x3v = CellCenterX(k - indcs.ks, indcs.nx3, size.d_view(m).x3min, size.d_view(m).x3max);

    const Real nu    = disk.Viscosity(x1v, x2v, x3v, t_now, binary_position);
    const Real eta   = rho * nu;
    const Real coeff = 1.0 / 3.0;

    const Real ax = eta * (lap_vx + coeff * grad_div_x);
    const Real ay = eta * (lap_vy + coeff * grad_div_y);
    const Real az = eta * (lap_vz + coeff * grad_div_z);

    u(m, IVX, k, j, i) += beta_dt_local * ax;
    u(m, IVY, k, j, i) += beta_dt_local * ay;
    u(m, IVZ, k, j, i) += beta_dt_local * az;

    if (update_energy) {
      const Real visc_work = vx * ax + vy * ay + vz * az;
      u(m, IEN, k, j, i) += beta_dt_local * visc_work;
    }
  });
}

// 2. Point mass gravity and accretion 
KOKKOS_INLINE_FUNCTION
void point_mass_source_term(const Kokkos::Array<Kokkos::Array<Real, 3>, 2> &binary_velocity,
                            const Kokkos::Array<Kokkos::Array<Real, 3>, 2> &binary_position,
                            const Real x,
                            const Real y,
                            const Real z,
                            const Real dt,
                            const Real rho,
                            const Real vx,
                            const Real vy,
                            const Real vz,
                            const int which_mass,
                            Real delta_cons[5]) {
    for (int n = 0; n < 5; ++n) {
      delta_cons[n] = 0.0;
    }

    if (dt <= 0.0 || rho <= 0.0) {
      return;
    }

    const bool primary          = (which_mass == 1);
    const Real sink_radius      = primary ? g_binary->rsink1() : g_binary->rsink2();
    const Real softening_radius = primary ? g_binary->rsoft1() : g_binary->rsoft2();
    const Real mass             = primary ? g_binary->m1()     : g_binary->m2();
    const auto &pos             = binary_position[which_mass - 1];
    const auto &vel             = binary_velocity[which_mass - 1];

    const Real x0 = pos[0];
    const Real y0 = pos[1];
    const Real z0 = pos[2];
    const Real dx = x - x0;
    const Real dy = y - y0;
    const Real dz = z - z0;
    const Real r2 = dx * dx + dy * dy + dz * dz;
    const Real dr = Kokkos::sqrt(r2);

    Real sink_rate = 0.0;
    if (sink_radius > 0.0 && dr < 2.0 * sink_radius) {
      const Real ratio = dr / sink_radius;
      sink_rate = g_binary->SinkRate() * Kokkos::exp(-Kokkos::pow(ratio, 4.0));
    }
    if (dt > 0.0 && sink_rate > 0.0) {
      sink_rate = Kokkos::min(sink_rate, 0.9 / dt);
    }

    const Real fgrav_numerator = rho * mass * Kokkos::pow(r2 + softening_radius * softening_radius, -1.5);
    const Real fx              = -fgrav_numerator * dx;
    const Real fy              = -fgrav_numerator * dy;
    const Real fz              = -fgrav_numerator * dz;
    const Real mdot            = -rho * sink_rate;

    if (g_binary->SinkModel() == 0) {
      return;
    }

    if (g_binary->SinkModel() == 1) {
      // acceleration free
      delta_cons[0] = dt * mdot;
      delta_cons[1] = dt * (mdot * vx) + dt * fx;
      delta_cons[2] = dt * (mdot * vy) + dt * fy;
      delta_cons[3] = dt * (mdot * vz) + dt * fz;
      delta_cons[4] = 0.0;
      return;
    }

    if (g_binary->SinkModel() == 2) {
      // torque free
      const Real vx0       = vel[0];
      const Real vy0       = vel[1];
      const Real vz0       = vel[2];
      const Real inv_dr    = 1.0 / (dr + 1e-12);
      const Real rhatx     = dx * inv_dr;
      const Real rhaty     = dy * inv_dr;
      const Real rhatz     = dz * inv_dr;
      const Real dvdotrhat = (vx - vx0) * rhatx + (vy - vy0) * rhaty + (vz - vz0) * rhatz;
      const Real vxstar    = dvdotrhat * rhatx + vx0;
      const Real vystar    = dvdotrhat * rhaty + vy0;
      const Real vzstar    = dvdotrhat * rhatz + vz0;

      delta_cons[0] = dt * mdot;
      delta_cons[1] = dt * (mdot * vxstar) + dt * fx;
      delta_cons[2] = dt * (mdot * vystar) + dt * fy;
      delta_cons[3] = dt * (mdot * vzstar) + dt * fz;
      delta_cons[4] = 0.0;
      return;
    }
}


void binary_source_term(Mesh *pm, const Real beta_dt, Kokkos::Array<Kokkos::Array<Real, 3>, 2> binary_position, Kokkos::Array<Kokkos::Array<Real, 3>, 2> binary_velocity) {
  if (!g_sources_enabled || beta_dt <= 0.0) {
    return;
  }

  if (pm == nullptr || pm->pmb_pack == nullptr || g_binary == nullptr) {
    return;
  }

  MeshBlockPack *pmbp = pm->pmb_pack;
  if (pmbp->phydro == nullptr) {
    return;
  }

  //const Binary binary      = *binary;
  auto &indcs              = pm->mb_indcs;
  auto &size               = pmbp->pmb->mb_size;
  auto &w                  = pmbp->phydro->w0;
  auto &u                  = pmbp->phydro->u0;
  const bool update_energy = pmbp->phydro->peos->eos_data.is_ideal;
  const Real beta_dt_local = beta_dt;

  par_for("binary_sink_source",
          DevExeSpace(),
          0, (pmbp->nmb_thispack - 1),
          indcs.ks, indcs.ke,
          indcs.js, indcs.je,
          indcs.is, indcs.ie,
          KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const Real x1v = CellCenterX(i - indcs.is, indcs.nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    const Real x2v = CellCenterX(j - indcs.js, indcs.nx2, size.d_view(m).x2min, size.d_view(m).x2max);
    const Real x3v = CellCenterX(k - indcs.ks, indcs.nx3, size.d_view(m).x3min, size.d_view(m).x3max);

    const Real rho = w(m, IDN, k, j, i);
    const Real vx  = w(m, IVX, k, j, i);
    const Real vy  = w(m, IVY, k, j, i);
    const Real vz  = w(m, IVZ, k, j, i);

    Real delta_total[5] = {0.0, 0.0, 0.0, 0.0, 0.0};

    for (int which_mass = 1; which_mass <= 2; ++which_mass) {
      Real delta_cons[5];
      point_mass_source_term(binary_velocity,
                             binary_position,
                             x1v,
                             x2v,
                             x3v,
                             beta_dt_local,
                             rho,
                             vx,
                             vy,
                             vz,
                             which_mass,
                             delta_cons);
      for (int n = 0; n < 5; ++n) {
        delta_total[n] += delta_cons[n];
      }
    }

    u(m, IDN, k, j, i) += delta_total[0];
    u(m, IVX, k, j, i) += delta_total[1];
    u(m, IVY, k, j, i) += delta_total[2];
    u(m, IVZ, k, j, i) += delta_total[3];
    if (update_energy) {
      u(m, IEN, k, j, i) += delta_total[4];
    }
  });
}




// 3. Locally isothermal equation of state and setting sound speed
void enforce_isothermal_pressure(Mesh *pm) {
  if (!g_sources_enabled) {
    return;
  }

  if (pm == nullptr || pm->pmb_pack == nullptr || g_binary == nullptr) {
    return;
  }

  MeshBlockPack *pmbp = pm->pmb_pack;
  if (pmbp->phydro == nullptr) {
    return;
  }

  // const Disk disk(g_mach, g_alpha, g_eps, *g_binary);
  const auto binary_position = g_binary->Position(pm->time);

  auto &indcs              = pm->mb_indcs;
  auto &size               = pmbp->pmb->mb_size;
  auto &w                  = pmbp->phydro->w0;
  auto &u                  = pmbp->phydro->u0;
  const bool update_energy = pmbp->phydro->peos->eos_data.is_ideal;
  const Real gm1           = update_energy ? (pmbp->phydro->peos->eos_data.gamma - 1.0) : 0.0;

  par_for("enforce_isothermal_pressure",
          DevExeSpace(),
          0, (pmbp->nmb_thispack - 1),
          indcs.ks, indcs.ke,
          indcs.js, indcs.je,
          indcs.is, indcs.ie,
          KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const Real x1v = CellCenterX(i - indcs.is, indcs.nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    const Real x2v = CellCenterX(j - indcs.js, indcs.nx2, size.d_view(m).x2min, size.d_view(m).x2max);
    const Real x3v = CellCenterX(k - indcs.ks, indcs.nx3, size.d_view(m).x3min, size.d_view(m).x3max);

    const Real rho      = w(m, IDN, k, j, i);
    const Real cs2      = g_disk->SoundSpeedSquare(x1v, x2v, x3v, binary_position);
    const Real pressure = rho * cs2;
    w(m, IPR, k, j, i)  = pressure;

    if (update_energy) {
      const Real vx      = w(m, IVX, k, j, i);
      const Real vy      = w(m, IVY, k, j, i);
      const Real vz      = w(m, IVZ, k, j, i);
      const Real kinetic = 0.5 * rho * (vx * vx + vy * vy + vz * vz);
      u(m, IEN, k, j, i) = pressure / gm1 + kinetic;
    }
  });
}


// 4. Set floors on density and pressure, and handle NaN values
void enforce_floors(Mesh *pm) {

  MeshBlockPack *pmbp = pm->pmb_pack;
  if (pmbp->phydro == nullptr) {
    return;
  }

  auto &indcs              = pm->mb_indcs;
  auto &size               = pmbp->pmb->mb_size;
  auto &w                  = pmbp->phydro->w0;
  auto &u                  = pmbp->phydro->u0;

  par_for("enforce_floors",
          DevExeSpace(),
          0, (pmbp->nmb_thispack - 1),
          indcs.ks, indcs.ke,
          indcs.js, indcs.je,
          indcs.is, indcs.ie,
          KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
          

      // apply floors over Nan values
      if ((Kokkos::isnan(w(m, IDN, k, j, i))) || Kokkos::isnan(w(m, IPR, k, j, i))) {
          w(m, IDN, k, j, i) = g_floor_density;
          u(m, IDN, k, j, i) = g_floor_density;
      }

      // enfore the floors
      bool update_cons = false;
      if (w(m, IDN, k, j, i) < g_floor_density)  {
        update_cons        = true;
        w(m, IDN, k, j, i) = g_floor_density;
        w(m, IVX, k, j, i) = 0.0;  
        w(m, IVY, k, j, i) = 0.0;
        w(m, IVZ, k, j, i) = 0.0;
      }
      if (w(m, IPR, k, j, i) < g_floor_pressure) {w(m, IPR, k, j, i) = g_floor_pressure; update_cons = true;}

      if (update_cons) {
        u(m, IDN, k, j, i) = w(m, IDN, k, j, i);
        u(m, IM1, k, j, i) = w(m, IVX, k, j, i) * w(m, IDN, k, j, i);
        u(m, IM2, k, j, i) = w(m, IVY, k, j, i) * w(m, IDN, k, j, i);
        u(m, IM3, k, j, i) = w(m, IVZ, k, j, i) * w(m, IDN, k, j, i);
        u(m, IEN, k, j, i) = w(m, IPR, k, j, i) / (pmbp->phydro->peos->eos_data.gamma - 1.0);
      }
    });
}


//----------------------------------------------------------------------------------------
// (OPTIONAL) Custom boundary conditions
// Uncomment and implement if needed (e.g., for outflow at boundaries)
//----------------------------------------------------------------------------------------
/*
void CircumbinaryUserBCs(Mesh* pm) {
  // Custom boundary condition implementation
  // This is called each timestep by ApplyPhysicalBCs in the task list
  // Example: set outflow boundary conditions at disk edges
}
*/

//----------------------------------------------------------------------------------------
// (OPTIONAL) Custom history/diagnostics output
// Uncomment and implement if needed (e.g., to track disk mass, angular momentum)
//----------------------------------------------------------------------------------------
/*
void CircumbinaryHistory(HistoryData *pdata, Mesh *pm) {
  // Compute and store custom diagnostics
  // Example quantities:
  // - Total disk mass
  // - Angular momentum
  // - Accretion rate
  // - Temperature statistics
}
*/

//----------------------------------------------------------------------------------------
// (OPTIONAL) Final analysis after simulation
// Uncomment and implement if needed (e.g., error computation, final diagnostics)
//----------------------------------------------------------------------------------------
/*
void CircumbinaryFinalAnalysis(ParameterInput *pin, Mesh *pm) {
  // Post-simulation analysis
  // Called in Driver::Finalize()
}
*/
