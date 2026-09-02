# =====================================================================
# The assembled fluid model and the ROM it produces.
# =====================================================================

"""
	AssembledFluidModel <: AbstractAssembledModel

Incompressible Navier-Stokes about a steady base flow, assembled and ready for a
reduction: the FE spaces and DOF maps, the base flow itself, the linearised
operators, and the Reynolds-continuation pieces.

The system is **first order and descriptor** — `B₁ ṡ = −B₀ s + …` with `B₁`
singular, because pressure carries no time derivative. So `ORD = 1`: there are no
companion derivative blocks anywhere downstream, which is why
[`build_model`](@ref) can use the plain-matrix `SpectralData` constructor.

# Fields
- `fom`         — the `setup_fem` bundle: grids, `CellValues`, DOF ranges, free-DOF maps
- `Re₀`         — the Reynolds number the base flow and operators were built at
- `s₀_full`     — Newton base flow over **all** DOFs (prescribed inlet included)
- `B`           — the linear operators `(B₀, B₁)`, free × free: the linearised
  operator and the (singular) mass. See the indexing note below.
- `K_visc`      — viscous coupling, free × free, **already scaled by `−D`**
- `K_visc_rect` — the same operator free × ALL, kept because `h₀` needs the
  prescribed inlet DOFs that the square block drops
- `h₀_vec`      — base-flow forcing direction, **already scaled by `−D`**
- `info`        — DOF counts and per-stage timings

## Indexing `B`

`B` is a plain 1-indexed `Tuple`, so **`B[k + 1]` is the mathematics' `Bₖ`**:

	m.B[1]   # B₀, the linearised operator
	m.B[2]   # B₁, the singular mass

That is the same convention as `MORFE.NthOrderModel.linear_terms`, which `B` is handed
to verbatim — `linear_terms[1]` is `B₀` there too. Grouping the operators also means
their order is defined in exactly one place instead of being implied by the order the
fields happen to be declared in.

`m.B₀` and `m.B₁` keep working as read-only properties. They are the older spelling,
retained because `examples/11_parametric_karman_profile` reads them in sixteen places;
new code should use `m.B[1]` / `m.B[2]`.

`K_visc` and `h₀_vec` arrive pre-scaled deliberately: the `−D` factor (`D` the
cylinder diameter, the reference length in `ν = D/Re`) used to be applied by the
example driver, which meant the convention lived in a comment and depended on two
copies of `D` agreeing. It is applied once, here.
"""
struct AssembledFluidModel{F, TB, TK, TR} <: AbstractAssembledModel
	fom::F
	Re₀::Float64
	s₀_full::Vector{Float64}
	B::TB
	K_visc::TK
	K_visc_rect::TR
	h₀_vec::Vector{Float64}
	info::NamedTuple
end

# The pre-`B` spelling. `getproperty` rather than a deprecation because these are read in
# a research example whose numerics must not be disturbed by this rename.
function Base.getproperty(m::AssembledFluidModel, s::Symbol)
	s === :B₀ && return getfield(m, :B)[1]
	s === :B₁ && return getfield(m, :B)[2]
	return getfield(m, s)
end

Base.propertynames(::AssembledFluidModel, private::Bool = false) = (:fom, :Re₀, :s₀_full,
	:B, :K_visc, :K_visc_rect, :h₀_vec, :info, :B₀, :B₁)

function Base.show(io::IO, ::MIME"text/plain", m::AssembledFluidModel)
	println(io, "AssembledFluidModel (Ferrite P2/P1 Taylor-Hood)")
	println(io, "  Re₀       : $(m.Re₀)")
	println(io, "  free DOFs : $(m.info.n_free) (steady state), $(m.info.n_free_dpim) (DPIM)")
	println(io, "  operators : B₀ nnz=$(nnz(m.B[1]))  B₁ nnz=$(nnz(m.B[2]))  K_visc nnz=$(nnz(m.K_visc))")
	print(io, "  base flow : ‖s₀‖∞ = $(maximum(abs, m.s₀_full))")
end

# The rows only this physics has. The shared skeleton (sizes, order, stage timings)
# comes from Common.write_summary and is not restated here.
#
# The second argument is `build_model`'s own `meta` — there is no ROM wrapper to read,
# because there is no backend `parametrise` to build one. Everything needed is already
# in `meta`, which is the point of returning it.
function Common.summary_entries(m::AssembledFluidModel, meta::NamedTuple)
	rows = Pair{String, Any}[
		"model" => "2D Navier-Stokes, Ferrite P2/P1 Taylor-Hood",
		"Re0" => m.Re₀,
		"reference_length" => m.info.reference_length,
	]
	if haskey(meta, :spectrum)
		λ = collect(meta.spectrum.eigenvalues)
		push!(rows, "master_modes" => "$(length(λ))  (Hopf pair)")
		push!(rows, "master_eigenvalues" => λ)
	end
	# The gauge changes what W and R MEAN, so it belongs in the record of the run and
	# not only in the code that made it: coefficients from two gauges are not comparable.
	haskey(meta, :scale) && push!(rows, "mode_scale" => meta.scale)
	haskey(meta, :normalisation) &&
		push!(rows, "mode_normalisation" => nameof(typeof(meta.normalisation)))
	haskey(meta, :conjugate_permutation) &&
		push!(rows, "conjugate_permutation" => meta.conjugate_permutation)
	return rows
end

Common.summary_entries(m::AssembledFluidModel, ::Nothing) =
	Pair{String, Any}["model" => "2D Navier-Stokes, Ferrite P2/P1 Taylor-Hood",
		"Re0" => m.Re₀]
