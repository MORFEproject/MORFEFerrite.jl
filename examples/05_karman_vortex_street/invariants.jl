# invariants.jl — the physical quantities that identify this ROM, computed so that
# they DO NOT depend on how many modes were promoted into the master set.
#
# Why this exists. Raw R coefficients are not comparable between two runs:
#   · the Arpack eigenvector gauge differs run to run (z → e^{iφ}z), and
#   · promoting resonant outer modes into the master set is a CHANGE OF COORDINATES —
#     the ROM gains columns and shares no monomials with an un-promoted one.
# Both runs nonetheless describe the same physics, so the comparison has to be made on
# quantities that survive both transformations. Those are the ones below.
#
# The trick for promotion-invariance is SLAVING, exactly as solver/rom_palc.jl does it:
# on the slow manifold the promoted coordinates are quasi-steady, so they solve
# R_k(ρ, ρ, y, η) = 0. The monomial set carries at most one promoted coordinate to the
# first power, so R is exactly AFFINE in y — one small linear solve, not an iteration
# and not an approximation. Zeroing them instead would drop the mean-flow distortion,
# which is the dominant stabilising term in the Landau coefficient, and would report a
# supercritical Hopf as subcritical.

using Printf: @printf, @sprintf
using DelimitedFiles: readdlm

"Coordinate layout produced by main.jl: z₁, z̄₁, then any promoted modes y, then η′ last."
struct ROMPoly
	exps::Vector{Vector{Int}}      # monomial exponents, one entry per row
	coef::Matrix{ComplexF64}       # nmono × ncomp
	nvar::Int
end

n_promoted(p::ROMPoly) = p.nvar - 3

"""
    load_rom_poly(csv) → ROMPoly

Read an `R_coefficients.csv` (`exp_1…exp_NVAR, R1_re, R1_im, R2_re, …`). NVAR is
inferred from the header, so this reads a promoted and an un-promoted run alike.
"""
function load_rom_poly(csv::AbstractString)
	raw, hdr = readdlm(csv, ','; header = true)
	cols = vec(String.(hdr))
	nvar = count(startswith("exp_"), cols)
	ncomp = (length(cols) - nvar) ÷ 2
	exps = [Int[raw[i, j] for j in 1:nvar] for i in 1:size(raw, 1)]
	coef = Matrix{ComplexF64}(undef, size(raw, 1), ncomp)
	for k in 1:ncomp
		coef[:, k] = complex.(raw[:, nvar + 2k - 1], raw[:, nvar + 2k])
	end
	return ROMPoly(exps, coef, nvar)
end

function eval_rom(p::ROMPoly, z::AbstractVector{ComplexF64})
	out = zeros(ComplexF64, size(p.coef, 2))
	@inbounds for i in eachindex(p.exps)
		m = one(ComplexF64)
		e = p.exps[i]
		for j in 1:(p.nvar)
			ej = e[j]
			ej == 0 || (m *= z[j]^ej)
		end
		iszero(m) && continue
		for k in eachindex(out)
			out[k] += p.coef[i, k] * m
		end
	end
	return out
end

"""
    slaved_R1(p, ρ, η) → ComplexF64

First component of the reduced dynamics at the canonical phase z₁ = z̄₁ = ρ, η′ = η,
with every promoted coordinate slaved to its quasi-steady value. Identical in intent to
`solver/rom_palc.jl`'s `_rom_R1`, and reduces to a plain evaluation when nothing was
promoted.
"""
function slaved_R1(p::ROMPoly, ρ::Float64, η::Float64)
	z = zeros(ComplexF64, p.nvar)
	z[1] = ρ
	z[2] = ρ
	z[end] = η
	npro = n_promoted(p)
	npro == 0 && return eval_rom(p, z)[1]

	rows = 3:(2 + npro)
	b = ComplexF64[eval_rom(p, z)[k] for k in rows]      # R_k at y = 0
	A = Matrix{ComplexF64}(undef, npro, npro)
	for j in 1:npro                                      # ∂R_k/∂y_j, exact (R is affine in y)
		zj = copy(z)
		zj[2 + j] = 1
		rj = eval_rom(p, zj)
		for (k, kk) in enumerate(rows)
			A[k, j] = rj[kk] - b[k]
		end
	end
	z[rows] .= A \ (-b)
	return eval_rom(p, z)[1]
end

# R₁(ρ, ρ, η)/ρ is even in ρ (phase symmetry z → e^{iφ}z), so it expands as
# λ(η) + c₂₁₀(η)ρ² + O(ρ⁴). One Richardson step in ρ² therefore removes the ρ⁴ term
# exactly rather than merely reducing it.
function _lambda_and_c210(p::ROMPoly, η::Float64; ρ::Float64 = 1e-3)
	coarse = slaved_R1(p, ρ, η) / ρ
	fine = slaved_R1(p, ρ / 2, η) / (ρ / 2)
	# NOTE the explicit `*`. `4fine` would be fine, but `4f2` — the name this once
	# carried — parses as the Float32 literal 4e2 = 400.0, because `f` is Julia's
	# Float32 exponent marker. It silently returned λ = 133.33 instead of 16.86i.
	λ = (4 * fine - coarse) / 3            # Richardson in ρ²: kills the ρ² term
	c210 = (coarse - λ) / ρ^2
	return λ, c210
end

"""
    rom_invariants(p; η_step) → NamedTuple

The promotion- and gauge-aware summary of a ROM:

  · `σ`, `ω`      — the Hopf eigenvalue λ = σ + iω at η′ = 0. Fully invariant.
  · `c101`        — ∂λ/∂η′, the linear Reynolds sensitivity. Fully invariant. Computed
                    by a central difference, because in a PROMOTED run the mean-flow
                    coordinates respond to η′ directly and feed back through z₁·y_k, so
                    reading the `[1,0,1]` coefficient alone would miss part of it.
  · `c210_eff`    — the effective Landau coefficient after slaving. Scales as |c|² under
                    z → cz, so it is comparable only at a fixed `MODE_SCALE`; recorded
                    for completeness and checked loosely.
  · `c210_ratio`  — Im(c₂₁₀)/Re(c₂₁₀). The |c|² cancels, so this is the gauge-free
                    fingerprint of the nonlinearity — and the quantity whose SIGN the
                    conjugate-pairing bug flipped.
  · `criticality` — sign(Re c₂₁₀): supercritical (negative) or subcritical.
"""
function rom_invariants(p::ROMPoly; η_step::Float64 = 1e-4)
	λ, c210 = _lambda_and_c210(p, 0.0)
	# λ(η) is a polynomial in η, so a central difference leaves an O(η²) error that the
	# reference's large c₁₀₂ ≈ 2.9e3 makes visible (3e-6 relative at η = 1e-4). One
	# Richardson step in η² removes it, taking c₁₀₁ to ~1e-10 for four extra evaluations.
	d_coarse = (first(_lambda_and_c210(p, η_step)) -
				first(_lambda_and_c210(p, -η_step))) / (2η_step)
	d_fine = (first(_lambda_and_c210(p, η_step / 2)) -
			  first(_lambda_and_c210(p, -η_step / 2))) / η_step
	c101 = (4 * d_fine - d_coarse) / 3
	return (; σ = real(λ), ω = imag(λ),
		c101_re = real(c101), c101_im = imag(c101),
		c210_re = real(c210), c210_im = imag(c210),
		c210_ratio = imag(c210) / real(c210),
		criticality = real(c210) < 0 ? "supercritical" : "subcritical",
		nvar = p.nvar, n_promoted = n_promoted(p))
end

# ── Reference file I/O ────────────────────────────────────────────────────────
# Flat `key = value` text: diffable, greppable, and readable without a TOML dependency.

const INVARIANT_KEYS = ("σ", "ω", "c101_re", "c101_im",
	"c210_re", "c210_im", "c210_ratio")

function write_invariants(path::AbstractString, inv::NamedTuple; note::AbstractString = "")
	open(path, "w") do io
		println(io, "# Kármán vortex street — promotion-invariant ROM reference.")
		println(io, "# Valid for a run WITH or WITHOUT promoted outer modes: every quantity")
		println(io, "# here is computed after slaving the promoted coordinates.")
		isempty(note) || println(io, "# ", note)
		println(io)
		for k in INVARIANT_KEYS
			@printf(io, "%-12s = %.12g\n", k, getproperty(inv, Symbol(k)))
		end
		println(io)
		println(io, "# informational only — not compared")
		println(io, "criticality  = ", inv.criticality)
	end
	return path
end

function read_invariants(path::AbstractString)
	d = Dict{String, Float64}()
	for line in eachline(path)
		s = strip(first(split(line, '#')))
		isempty(s) && continue
		k, _, v = partition_kv(s)
		k in INVARIANT_KEYS && (d[k] = parse(Float64, v))
	end
	return d
end

function partition_kv(s::AbstractString)
	i = findfirst('=', s)
	i === nothing && return ("", "", "")
	return (strip(s[1:(i - 1)]), "=", strip(s[(i + 1):end]))
end
