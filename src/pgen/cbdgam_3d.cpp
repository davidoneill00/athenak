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
  static constexpr Real G      = 6.6725985e-8;
  static constexpr Real c      = 2.99792458e10;
	static constexpr Real kb     = 1.38065812e-16;
  static constexpr Real sigmab = 5.6705119e-5;
  static constexpr Real mp     = 1.6726e-24;
  static constexpr Real kappa  = 0.4;            
  static constexpr Real pc     = 3.085678e18;
  static constexpr Real msun   = 1.989e33;
};


class ShakuraSunyaevDisk {
public:
  ShakuraSunyaevDisk(Real central_mass_msun, Real length_scale_pc, Real mach_number_a,
                     Real alpha, Real gamma, Real target_accretion_rate)
    : _central_mass_msun(central_mass_msun),
      _length_scale_pc(length_scale_pc),
      _mach_number_a(mach_number_a),
      _alpha(alpha),
      _gamma(gamma),
      _target_accretion_rate(target_accretion_rate) {}

  // Computed properties (like Python @property)
  Real Mass_cgs() const { return _central_mass_msun * cgs::msun; }
  Real Length_cgs() const { return _length_scale_pc * cgs::pc; }
  Real GM_cgs() const { return cgs::G * Mass_cgs(); }
  Real Time_cgs() const { return sqrt(pow(Length_cgs(), 3) / GM_cgs()); }
  Real RSchwz() const { return 2.0 * GM_cgs() / (cgs::c * cgs::c); }
  
  Real EddingtonRate() const {
    Real eta = 0.1;
    return 4.0 * M_PI * GM_cgs() / cgs::kappa / cgs::c / eta;
}
  
  Real EddingtonFrac() const {
    Real mp4_kb4 = pow(cgs::mp, 4.0) / pow(cgs::kb, 4.0);
    Real f0 = 10.2604 * pow(mp4_kb4 * cgs::sigmab / cgs::kappa, 0.5) * pow(_gamma, -2.0);
    return f0 * pow(_alpha, 0.5) * pow(GM_cgs(), 7.0/4.0) * pow(Length_cgs(), -1.0/4.0) 
           * pow(_mach_number_a, -5.0) / EddingtonRate();
  }

  Real AccretionRate() const {
    return EddingtonFrac() * EddingtonRate();
  }

  Real Mdrop() const {
    return _target_accretion_rate / EddingtonFrac();
  }

  // vertically integrated density and pressure coefficients (in cgs)
  Real Surface_Density() const {
    Real s0 = 0.269274 * pow((pow(cgs::mp, 4.0) / pow(cgs::kb, 4.0) * cgs::sigmab / cgs::kappa), 0.2) * pow(_gamma, -0.8);
    return s0 * pow(_alpha, -0.8) * pow(GM_cgs(), 0.2) * pow(AccretionRate(), 0.6) * pow(Length_cgs(), -0.6);
  }

  Real Surface_Pressure() const {
    return 0.106103 / _gamma / _alpha * AccretionRate() * sqrt(GM_cgs()) * pow(Length_cgs(), -1.5);
  }

  Real Midplane_Temperature() const {
    Real t0 = 0.394035 * pow(cgs::mp * cgs::kappa / cgs::kb / cgs::sigmab, 0.2) * pow(_gamma, -0.2);
    return t0 * pow(_alpha, -0.2) * pow(GM_cgs(), 0.3) * pow(AccretionRate(), 0.4) * pow(Length_cgs(), -0.9);
  }

  // Code unit conversion coefficients
  Real Surface_Density_Coefficient() const {
    return Surface_Density() / (Mass_cgs() / (Length_cgs() * Length_cgs()));
  }

  Real Surface_Pressure_Coefficient() const {
    return Surface_Pressure() / (Mass_cgs() / (Time_cgs() * Time_cgs()));
  }

  // Radial profile methods (take radius r in code units)
  Real Surface_Density_Profile(Real r) const {
    return Surface_Density_Coefficient() * pow(r, -0.6);  // r^(-3/5)
  }

  Real Surface_Pressure_Profile(Real r) const {
    return Surface_Pressure_Coefficient() * pow(r, -1.5);  // r^(-3/2)
  }

  Real Mach_Profile(Real r) const {
    Real cs = sqrt(_gamma * (Surface_Pressure_Profile(r) / Surface_Density_Profile(r)));
    Real Omega = pow(r, -1.5);  // Keplerian angular velocity
    Real Hs = cs / Omega;        // Scale height
    return r / Hs;
  }

  Real Scale_Height(Real r) const {
    Real cs = sqrt(_gamma * (Surface_Pressure_Profile(r) / Surface_Density_Profile(r)));
    Real Omega = pow(r, -1.5);
    return cs / Omega;
  }

  Real Optical_Depth(Real r) const {
    return cgs::kappa * Surface_Density() * pow(r, -0.6);
  }

  Real Cooling_Coefficient() const {
    Real mp_code = cgs::mp / Mass_cgs();
    Real kb_code = cgs::kb / (Mass_cgs() * Length_cgs() * Length_cgs() / (Time_cgs() * Time_cgs()));
    Real kappa_code = cgs::kappa / (Length_cgs() * Length_cgs() / Mass_cgs());
    Real sigmab_code = cgs::sigmab / (Mass_cgs() / pow(Time_cgs(), 3.0));
    return 8.0 / 3.0 * sigmab_code / kappa_code * pow(mp_code / kb_code, 4.0) * pow(_gamma - 1.0, 4.0);
  }

  Real Mdot_at_r(Real r) const {
    Real cs = sqrt(_gamma * (Surface_Pressure_Profile(r) / Surface_Density_Profile(r)));
    Real Omega = pow(r, -1.5);
    Real Hs = cs / Omega;
    Real nu = _alpha * cs * Hs;
    return 3.0 * M_PI * Surface_Density_Profile(r) * nu;
  }

  Real Mdot_inf() const {
    return Mdot_at_r(1.0);
  }

  // Code unit conversions
  Real kb_code() const {
    return cgs::kb / (Mass_cgs() * Length_cgs() * Length_cgs() / (Time_cgs() * Time_cgs()));
  }

  Real sigmab_code() const {
    return cgs::sigmab / (Mass_cgs() / pow(Time_cgs(), 3.0));
  }

  Real mp_code() const {
    return cgs::mp / Mass_cgs();
  }

  Real kappa_code() const {
    return cgs::kappa / (Length_cgs() * Length_cgs() / Mass_cgs());
  }

private:
  Real _central_mass_msun;
  Real _length_scale_pc;
  Real _mach_number_a;
  Real _alpha;
  Real _gamma;
  Real _target_accretion_rate;
}; 


// hydro callers
Real BinaryPotential(Real x, Real y, Real z, OrbitalState binary) {
  Real G          = 1.0;  // Gravitational constant in code units
  Vector3D r1_vec = {x - binary.x1, y - binary.y1, z - binary.z1};
  Vector3D r2_vec = {x - binary.x2, y - binary.y2, z - binary.z2};
  Real phi1       = -G * binary.m1 / sqrt(r1_vec[0]*r1_vec[0] + r1_vec[1]*r1_vec[1] + r1_vec[2]*r1_vec[2] + binary.soft1*binary.soft1);
  Real phi2       = -G * binary.m2 / sqrt(r2_vec[0]*r2_vec[0] + r2_vec[1]*r2_vec[1] + r2_vec[2]*r2_vec[2] + binary.soft2*binary.soft2);

  return phi1 + phi2;
}

Real BinaryOmega(Real x, Real y, Real z, OrbitalState binary) {
  Real G          = 1.0;  // Gravitational constant in code units
  Vector3D r1_vec = {x - binary.x1, y - binary.y1, z - binary.z1};
  Vector3D r2_vec = {x - binary.x2, y - binary.y2, z - binary.z2};
  Real r1         = sqrt(r1_vec[0]*r1_vec[0] + r1_vec[1]*r1_vec[1] + r1_vec[2]*r1_vec[2]);
  Real r2         = sqrt(r2_vec[0]*r2_vec[0] + r2_vec[1]*r2_vec[1] + r2_vec[2]*r2_vec[2]);
  Real omega1     = sqrt(G * binary.m1 / (r1*r1*r1));
  Real omega2     = sqrt(G * binary.m2 / (r2*r2*r2));
  return omega1 + omega2;  
}

Real SoundSpeedSquare(Real x, Real y, Real z, OrbitalState binary, Real Mach) {
  Real cs2 = BinaryPotential(x, y, z, binary) / (Mach * Mach);
  return cs2;
}

Real Viscosity(Real x, Real y, Real z, OrbitalState binary, Real alpha, Real Mach) {
  Real cs    = sqrt(SoundSpeedSquare(x, y, z, binary, Mach));
  Real H     = cs / BinaryOmega(x, y, z, binary);  // Scale height H = cs / Omega
  return alpha * cs * H;
}

// initialisation functions
Real InitialDensity_XY(Real r, Real z, Real Mach, OrbitalState binary) {
  Real GM       = 1.0 * (binary.m1 + binary.m2);      
  Real rho0     = 1.0;                           // Density normalization
  Real h2       = 1.0 / Mach / Mach;             // Aspect ratio H/R
  Real cs2      = h2 * GM / r;
  Real vertical = exp((1/h2) * (1/sqrt(1.0 + z*z/r/r) - 1.0));  
  Real cavity   = (r > 2.5) ? 1.0 : 0.000001;    // Cavity inside binary (r < 2.5)
  return rho0 * pow(r, -1.5) * vertical * cavity;
}

Real InitialDensity(Real x, Real y, Real z, Real theta, OrbitalState binary, Real Mach) {
  Real r     = sqrt(x*x + y*y + z*z);
  Real r_cyl = r * cos(theta);
  Real z_cyl = r * sin(theta);
  return InitialDensity_XY(r_cyl, z_cyl, Mach, binary);
}

Real InitialPressure(Real x, Real y, Real z, Real theta, OrbitalState binary, Real Mach) {
  Real rho = InitialDensity(x, y, z, theta, binary, Mach);
  Real cs2 = SoundSpeedSquare(x, y, z, binary, Mach);
  return rho * cs2;
}

// binary motion and source terms












// write binary force and accretion 


//----------------------------------------------------------------------------------------
// Main problem generator for circumbinary disk
//----------------------------------------------------------------------------------------
void ProblemGenerator::CircumbinaryDisk(ParameterInput *pin, const bool restart) {
  // Skip if this is a restart - initial conditions already in file
  if (restart) return;

  // ========== READ PARAMETERS FROM <problem> BLOCK ==========
  // Binary parameters
  //Real a_binary = pin->GetOrAddReal("problem", "a_binary", 1.0);
  Real q_mass       = pin->GetOrAddReal("problem", "q_mass"  , 1.0);
  Real r_soft       = pin->GetOrAddReal("problem", "r_smooth", 0.1);
  Real eccentricity = pin->GetOrAddReal("problem", "eccentricity", 0.0);
  //Real m_total  = pin->GetOrAddReal("problem", "m_total" , 1.0);

  // Disk parameters
  Real rout     = pin->GetOrAddReal("problem", "rout"    , 10.0);
  Real Mach     = pin->GetOrAddReal("problem", "Mach"    , 10.0);

  // Smoothing scale for Keplerian velocity (avoids singularity at center)
  
  Real alpha    = pin->GetOrAddReal("problem", "alpha"   , 0.01);


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
