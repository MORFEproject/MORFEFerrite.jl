(
	label = "mode_1_order_7_cnf",
	check = true, # true → pause after Phase 1 (resonance check) and ask [y/N]
	neig = 3,
	phys_modes = [1],
	max_degree = 3,
	forces = [
	# (frequency_mode = 3, shape_mode = 3, amplitude = 3),
	(frequency_mode = 1, shape_mode = 1, amplitude = 3),
	],
	# Q[omega] := 1/(rayleigh_α/omega + rayleigh_β*omega)
	# rayleigh_α = omega/Q[omega]
	rayleigh_α = 72.5864/100,# C = rayleigh_α*M
	rayleigh_β = 0.0,# C = rayleigh_β*K
	resonance = (
		style = :cnf,   # :cnf | :rnf | :graph
		tolerance_rel = 0.05,   # relative: resonant if |λⱼ - s| < tol_rel·|λⱼ|
	),
)
