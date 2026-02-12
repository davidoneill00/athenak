cgs = dict(
		G = 6.6725985e-8,
		c = 2.99792458e10,
		kb = 1.38065812e-16,
		sigmab = 5.6705119e-5,
		mp = 1.6726e-24,
		kappa = 0.4,            # electron scattering
		pc = 3.085678e18,
		msun = 1.989e33,
		h = 6.62607015e-27,
		blackbodyconst1 = 1.4745e-47,
		blackbodyconst2 = 4.79921e-11,
		c2h3 = 2.61463e-58,
		h_over_kb = 4.79921e-11,
		year=31556952,
		ev = 1.60218e-12
	)

logger = getLogger(__name__)

class ShakuraSunyaevDisk(NamedTuple):
	"""The central mass, length scale (e.g. binary separation), alpha, and mach number (at r=a)
	   for a Shakura-Sunyaev alpha-disk (Frank, King, & Raine 2002) 
		- for numerical reasons the disk Mach number at r=a is supplied (as oppoed to the
		  accretion rate), and mdot as a fraction of the eddington rate is determined 
		  accordingly.
		- the target_accretion_rate is supplied to package the accretion rate remapping together
		  with the disk structure as a single container and for clarity in checkpoints.
	"""

	central_mass_msun     : float
	length_scale_pc       : float
	mach_number_a         : float
	alpha                 : float
	gamma                 : float
	target_accretion_rate : float

	# -------------------------------------------------------------------------
	@property
	def _mass(self) -> float:
		return self.central_mass_msun * cgs['msun']

	@property
	def _length(self) -> float:
		return self.length_scale_pc * cgs['pc']
	
	@property
	def _time(self) -> float:
		return sqrt(self._length**3 / self._GM)  # Omega_b^-1

	@property
	def _GM(self) -> float:
		return self._mass * cgs['G']	
	
	@property
	def _rschwz(self):
		return 2 * self._GM / cgs['c']**2

	@property
	def _accretion_efficiency(self):
		"""
		For a compact object; eta = 0.1, 0.15
		"""
		return 0.1
	
	@property
	def _eddington_rate(self) -> float:
		"""
		Here: kappa = thompson / mass proton
		"""
		return 4 * pi * self._GM / cgs['kappa'] / cgs['c'] / self._accretion_efficiency


	@property
	def _eddington_fraction(self) -> float:
		"""
		Calculate the eddington fraction of a reference Shakura Sunyaev disk at r=a. 

		   f_edd = √(32𝛑^2/3) * (mp^4 / kb^4 * sigmab / kappa * alpha)^(1/2) * (GM)^(7/4) * Mach(r)^-5 * r^(-1/4) / Mdot_edd

		This fraction of the eddington rate is returned
		"""
		f0 = 10.2604 * (cgs['mp']**4 / cgs['kb']**4 * cgs['sigmab'] / cgs['kappa'])**0.5 * self.gamma**(-2.)
		return f0 * self.alpha**0.5 * self._GM**(7./4.) * self._length**(-1./4.) * self.mach_number_a**-5 / self._eddington_rate
	
	@property
	def _accretion_rate(self) -> float:
		"""
		The actual accretion rate, parameterised by the Eddington rate
		"""
		return self._eddington_fraction * self._eddington_rate
	
	@property
	def Mdrop(self):
		return self.target_accretion_rate / self._eddington_fraction

	@property
	def _surface_density(self) -> float:
		"""
		Disk surface density from Shakura Sunyaev in cgs

		   Sigma = (32 * 3^6 / pi^3)^(1/5) * (mp^4 / kb^4 * sigmab / kappa)^(1/5) 
		   			* alpha^(-4/5) * (GM)^(1/5) * Mdot^(3/5) * r^(-3/5)
		"""
		s0 = 0.269274 * (cgs['mp']**4 / cgs['kb']**4 * cgs['sigmab'] / cgs['kappa'])**(1./5.) * self.gamma**(-4./5.)
		return s0 * self.alpha**(-4./5.) * self._GM**(1./5.) * self._accretion_rate**(3./5.) * self._length**(-3./5.)

	@property
	def _surface_pressure(self) -> float:
		"""Disk pressure scaling in cgs

		   P = (1 / 3 / pi) * alpha^-1 * Mdot * (GM)^0.5 * r^(-3/2)
		"""
		return 0.106103 / self.gamma / self.alpha * self._accretion_rate * sqrt(self._GM) * self._length**(-3./2.)

	@property
	def _midplane_temperature(self) -> float:
		"""Disk midplane temperature scaling in cgs (for completeness)

		   T = (3 / 32 / pi^2)^(1/5) * (mp * kappa / kb / sigmab)^(1/5) 
		        * alpha^(-1/5) * (GM)^(3/10) * Mdot^(2/5) * r^(-9/10)
		"""
		t0 = 0.394035 * (cgs['mp'] * cgs['kappa'] / cgs['kb'] / cgs['sigmab'])**(1./5.) * self.gamma**(-1./5.)
		return t0 * self.alpha**(-1./5.) * self._GM**(3./10.) * self._accretion_rate**(2./5.) * self._length**(-9./10.)

	# -------------------------------------------------------------------------
	@property
	def surface_density_coefficient(self) -> float:
		return self._surface_density / (self._mass / self._length**2)

	@property
	def surface_pressure_coefficient(self) -> float:
		return self._surface_pressure / (self._mass / self._time**2)	

	def surface_density_profile(self, r:float) -> float:
		return self.surface_density_coefficient * r**(-3./5.)

	def surface_pressure_profile(self, r:float) -> float:
		return self.surface_pressure_coefficient * r**(-3./2.)

	def mach_profile(self, r:float) -> float:
		cs    = (self.gamma * (self.surface_pressure_profile(r) / self.surface_density_profile(r)))**0.5
		Omega = r**(-3./2.)
		Hs    = cs /Omega
		return r / Hs

	def optical_depth(self, r:float) -> float:
		return cgs['kappa'] * self._surface_density * r**(-3./5.)

	# -------------------------------------------------------------------------
	@property
	def cooling_coefficient(self) -> float:
		"""
		Assumes avg fluid particle mass is the proton mass
	
				P / Sigma = eps * (gamma - 1)
	
				deps/dt = - Qdot / Sigma  & Qdot = 8 / 3 * sigma_boltzmann / opacity / Sigma * T^4
	
				eps_cooled (dt) = eps * (1 + 3 * cooling_coefficient * Sigma^-2 * eps^3 * dt)^-1/3 
	
				cooling_coefficient = 8/3 * sigma_boltz / opacity * (mp / kb) * (gamma - 1)
		"""
		mp_code = cgs['mp'] /  self._mass
		kb_code = cgs['kb'] / (self._mass * self._length**2 / self._time**2)
		kappa_code  = cgs['kappa'] / (self._length**2 / self._mass)
		sigmab_code = cgs['sigmab'] / (self._mass / self._time**3)	
		qdot_coeff = 8. / 3. * sigmab_code / kappa_code * (mp_code / kb_code)**4 * (self.gamma - 1.)**4
		return qdot_coeff

	# Only for temporary testing
	# =============================================================================
	def surface_density_goodman(self):
		coeff = 2**(4./5.) / 3. / pi**(3./5.)
		s0 = coeff * (cgs['mp']**4 / cgs['kb']**4 * cgs['sigmab'] / cgs['kappa'])**(1./5.)
		return s0 * self.alpha**(-4./5.) * self._GM**(1./5.) * self._accretion_rate**(3./5.) * self._length**(-3./5.)
		coeff = 2**(4./5.) / 3. / pi**(3./5.)
		s0 = coeff * (cgs['mp']**4 / cgs['kb']**4 * cgs['sigmab'] / cgs['kappa'])**(1./5.)
		return s0 * self.alpha**(-4./5.) * self._GM**(1./5.) * self._accretion_rate**(3./5.) * self._length**(-3./5.)

	def midplane_temperature_goodman(self):
		coeff = (1. / 16. / pi**2)**(1./5.)
		t0 = coeff * (cgs['mp'] * cgs['kappa'] / cgs['kb'] / cgs['sigmab'])**(1./5.)
		return t0 * self.alpha**(-1./5.) * self._GM**(3./10.) * self._accretion_rate**(2./5.) * self._length**(-9./10.)

	def surface_pressure_goodman(self):
		return cgs['kb'] / cgs['mp'] * self.midplane_temperature_goodman() * self.surface_density_goodman()
	
	def Mdot(self, r):
		cs    = (self.gamma * (self.surface_pressure_profile(r) / self.surface_density_profile(r)))**0.5
		Omega = r**(-3./2.)
		Hs    = cs / Omega
		nu    = self.alpha * cs * Hs
		return 3 * pi * self.surface_density_profile(r) * nu
	
	@property
	def Mdot_inf(self):
		return self.Mdot(1.0)
		
	# =============================================================================
	# ========================== Code unit conversions ============================
	# =============================================================================

	@property
	def kb_code(self):
		return cgs['kb'] / (self._mass * self._length**2 / self._time**2)
	
	@property
	def sigmab_code(self):
		return cgs['sigmab'] / (self._mass / self._time**3)	

	@property
	def mp_code(self):
		return cgs['mp'] / (self._mass)

	@property
	def kappa_code(self):
		return cgs['kappa'] / (self._length**2 / self._mass)

	@property   
	def Length_Scale_CGS(self): # physical units (not code units)
		return self.length_scale_pc * cgs['pc']