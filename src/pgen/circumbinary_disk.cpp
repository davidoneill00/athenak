//! \file circumbinary_disk_custom.cpp
//! \brief Problem generator for a simple circumbinary accretion disk in 3D
//!
//! The disk is initialized with:
//! - Radial power-law density profile: rho ~ r^-alpha
//! - Vertical Gaussian structure: rho ~ exp(-z^2/2H^2)
//! - Keplerian rotation in the orbital plane
//! - Binary cavity


#include <math.h>
#include <algorithm>
#include <iostream>

#include "parameter_input.hpp"
#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "coordinates/cell_locations.hpp"


struct OrbitalState {
  Real m1, m2;
  Real x1, x2;
  Real y1, y2;
  Real z1, z2;
  Real vx1, vx2;
  Real vy1, vy2;  
  Real vz1, vz2;
  Real soft1, soft2;  
  Real eccentricity;
  Real semimajoraxis;
};

struct cgs {
  static constexpr Real G               = 6.6725985e-8,
  static constexpr Real c               = 2.99792458e10,
	static constexpr Real kb              = 1.38065812e-16,
  static constexpr Real sigmab          = 5.6705119e-5,
  static constexpr Real mp              = 1.6726e-24,
  static constexpr Real kappa           = 0.4,            
  static constexpr Real pc              = 3.085678e18,
  static constexpr Real msun            = 1.989e33,
  // static constexpr Real h               = 6.62607015e-27,
  // static constexpr Real blackbodyconst1 = 1.4745e-47,
  // static constexpr Real blackbodyconst2 = 4.79921e-11,
  // static constexpr Real c2h3            = 2.61463e-58,
  // static constexpr Real h_over_kb       = 4.79921e-11,
  // static constexpr Real year            = 31556952,
  // static constexpr Real ev              = 1.60218e-12
};

KOKKOS_INLINE_FUNCTION
Vector3D GetAngularMomentum(Real x, Real y, Real z, Real vx, Real vy, Real vz) {
  // L = r x v
  Real Lx = y*vz - z*vy;
  Real Ly = z*vx - x*vz;
  Real Lz = x*vy - y*vx;
  return {Lx, Ly, Lz};
}

KOKKOS_INLINE_FUNCTION
Vector3D GetNormalVector(Real x, Real y, Real z, Real vx, Real vy, Real vz) {
  Vector3D L = GetAngularMomentum(x, y, z, vx, vy, vz);
  Real L_mag = sqrt(L.x*L.x + L.y*L.y + L.z*L.z);
  return {L.x/L_mag, L.y/L_mag, L.z/L_mag};
}

KOKKOS_INLINE_FUNCTION
Real CylindricalRadius(Real x, Real y, Real z, Real vx, Real vy, Real vz) {
  Vector3D n      = GetNormalVector(x, y, z, vx, vy, vz);
  Vector3D r_proj = {n[0] * x, n[1] * y, n[2] * z};
  Real r_cyl      = sqrt(r_proj[0]*r_proj[0] + r_proj[1]*r_proj[1] + r_proj[2]*r_proj[2]);
  return r_cyl;
}

Real BinaryPotential(Real x, Real y, Real z, OrbitalState binary) {
  Real G          = 1.0;  // Gravitational constant in code units
  Vector3D r1_vec = {x - binary.x1, y - binary.y1, z - binary.z1};
  Vector3D r2_vec = {x - binary.x2, y - binary.y2, z - binary.z2};
  Real phi1       = -G * binary.m1 / sqrt(r1_vec[0]*r1_vec[0] + r1_vec[1]*r1_vec[1] + r1_vec[2]*r1_vec[2] + binary.soft1*binary.soft1);
  Real phi2       = -G * binary.m2 / sqrt(r2_vec[0]*r2_vec[0] + r2_vec[1]*r2_vec[1] + r2_vec[2]*r2_vec[2] + binary.soft2*binary.soft2);

  return phi1 + phi2;
}

// locally isothermal sound speed squared (evaluated at the mindplane)
Real MidplaneSoundSpeedSquare(Real x, Real y, Real z, OrbitalState binary, Real Mach) {
  Real cs_sq = BinaryPotential(x, y, z, binary) / (Mach * Mach);
  return cs_sq;
}

KOKKOS_INLINE_FUNCTION
Real VerticalFactor(Real d_perp, Real r_cyl, Real Mach) {
  Real h = 1.0 / Mach; 
  Real H = h * r_cyl;
  return exp(-0.5*SQR(d_perp/H));
}

// KOKKOS_INLINE_FUNCTION
// Real GetRadialDensity(Real r_cyl, Real rin, Real rout, Real rho_a, Real powerlaw) {  
//   Real rho = 0.0;  // vacuum outside disk
//   if (r_cyl >= rin && r_cyl <= rout) {
//     rho = rho_a * pow(r_cyl, powerlaw);
//   }
//   return rho;
// }


// Write function for density that takes angle and then gives initial density
Real InitialDensityProfile(Real r, Real d_perp, real Mach) {
  // 2D Shakura Sunyaev density profile
  
  // 2D density profile
  // Write rho_midplane = Sigma / H
  // Write vertical factor
  // Return rho_midplane * vertical_factor
  return rho;
}





//----------------------------------------------------------------------------------------
// Main problem generator for circumbinary disk
//----------------------------------------------------------------------------------------
void ProblemGenerator::CircumbinaryDisk(ParameterInput *pin, const bool restart) {
  // Skip if this is a restart - initial conditions already in file
  if (restart) return;

  // ========== READ PARAMETERS FROM <problem> BLOCK ==========
  // Binary parameters
  Real a_binary = pin->GetOrAddReal("problem", "a_binary", 1.0);
  Real q_mass   = pin->GetOrAddReal("problem", "q_mass"  , 1.0);
  Real m_total  = pin->GetOrAddReal("problem", "m_total" , 1.0);

  // Disk parameters
  Real rin      = pin->GetOrAddReal("problem", "rin"          , 0.5);
  Real rout     = pin->GetOrAddReal("problem", "rout"         , 10.0);
  Real h        = pin->GetOrAddReal("problem", "h"            , 0.1);
  //Real rho_disk = pin->GetOrAddReal("problem", "rho_disk", 1.0);
  Real rho_a    = pin->GetOrAddReal("problem", "rho_amb"      , 1.0);
  Real powerlaw = pin->GetOrAddReal("problem", "powerlaw"     , -1.5);  // rho ~ r^-alpha

  // Disk dynamics
  Real cs_sq              = pin->GetOrAddReal("problem", "cs_sq", 0.01);  // Sound speed squared
  Real v_radial_accretion = pin->GetOrAddReal("problem", "v_radial", 0.0);

  // Cavity/cutoff inside binary
  Real r_cavity = pin->GetOrAddReal("problem", "r_cavity", 0.5);

  // Smoothing scale for Keplerian velocity (avoids singularity at center)
  Real r_smooth = pin->GetOrAddReal("problem", "r_smooth", 0.1);

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
    Real gm1  = pmbp->phydro->peos->eos_data.gamma - 1.0;

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





      // ===== Convert to cylindrical coordinates =====
      Real r_cyl = sqrt(SQR(x1v) + SQR(x2v) + SQR(x3v));      // Cylindrical radius
      Real phi   = atan2(x2v, x1v);                           // Azimuthal angle
      Real z     = x3v;                                       // Height above/below disk

      // ===== Compute density =====
      Real rho = GetRadialDensity(r_cyl, rin, rout, rho_disk, rho_amb, alpha);
      rho     *= GetVerticalFactor(z, r_cyl, H_to_r);

      // Inside the binary cavity, set ambient density
      Real r_to_binary = r_cyl - 0.5*a_binary;  // Simple approximation of distance to binary
      if (r_cyl < r_cavity) {
        rho = rho_amb;
      }

      // ===== Compute velocity =====
      // Keplerian velocity in azimuthal direction
      Real v_phi = GetKeplerianVelocity(r_cyl, m_total, r_smooth);

      // Convert to Cartesian components
      // v_phi is tangential, so: v_x = -v_phi * sin(phi), v_y = v_phi * cos(phi)
      Real vx = -v_phi * sin(phi);
      Real vy = v_phi * cos(phi);
      
      // Optional: add radial inflow velocity (accretion)
      if (v_radial_accretion != 0.0) {
        vx += v_radial_accretion * cos(phi);
        vy += v_radial_accretion * sin(phi);
      }

      Real vz = 0.0;  // No vertical motion (can add if desired)

      // ===== Compute pressure =====
      // For isothermal disk: P = cs^2 * rho
      Real pres = cs_sq * rho;

      // ===== Set primitive variables =====
      w0_(m, IDN, k, j, i) = rho;              // Density
      w0_(m, IVX, k, j, i) = vx;               // Velocity x-component
      w0_(m, IVY, k, j, i) = vy;               // Velocity y-component
      w0_(m, IVZ, k, j, i) = vz;               // Velocity z-component
      w0_(m, IPR, k, j, i) = pres;             // Pressure

    });  // End par_for loop

    // ===== Convert primitives to conserved variables =====
    pmbp->phydro->peos->PrimToCons(w0_, pmbp->phydro->u0, is, ie, js, je, ks, ke);

  }  // End if phydro != nullptr

  // ========== OPTIONAL: USER-DEFINED FUNCTIONS ==========
  // Uncomment these if you add corresponding user function implementations:
  
  // pgen_final_func = &CircumbinaryFinalAnalysis;
  // user_bcs_func = &CircumbinaryUserBCs;
  // user_hist_func = &CircumbinaryHistory;

}  // End CircumbinaryDisk

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
