# =====================================================================
# Run summaries — one writer, extended per physics.
#
# Every backend wants to record the same skeleton after a run (what was solved,
# how big it was, how long each stage took) plus a handful of rows only that
# physics has (a Reynolds number, a forcing amplitude, a θ-basis). Writing a
# separate summary function per module duplicates the skeleton and lets the
# formats drift.
#
# So `write_summary` is a normal function that owns the skeleton, and
# `summary_entries` is the dispatch seam a physics module adds a method to —
# the same arrangement as MORFE's `print_setup`. A module that wants extra rows
# defines one short method; a module that wants none defines nothing.
# =====================================================================

using Printf: @printf, @sprintf

"""
	summary_entries(case, rom) -> Vector{Pair{String, Any}}

The physics-specific rows a backend contributes to its run summary, in the order
they should be written.

Add a method for your assembled-model type to extend the summary; the shared
skeleton (sizes, order, stage timings) is written by [`write_summary`](@ref) and
does not need restating. Returning an empty vector — the default — is fine.

```julia
function Common.summary_entries(m::AssembledFluidModel, rom::FluidROM)
    return ["Re0" => m.Re₀, "mode_scale" => rom.info.scale]
end
```

A row whose key repeats one the skeleton already wrote **overrides** it, so a
physics can also correct a shared row rather than only append to it.
"""
summary_entries(case, rom) = Pair{String, Any}[]

"""
	stage_timings(info) -> Vector{Pair{String, Float64}}

Every `*_time_s` entry of an `info` NamedTuple, in declaration order, with the
suffix stripped.

This is the convention that lets one writer time every backend without knowing
any of them: a stage that records `eig_time_s` in its `info` is reported as
"eig", and a backend adds a timed stage simply by recording it.
"""
function stage_timings(info)
	out = Pair{String, Float64}[]
	info === nothing && return out
	for k in keys(info)
		s = String(k)
		endswith(s, "_time_s") || continue
		v = getproperty(info, k)
		v isa Real && push!(out, s[1:(end - length("_time_s"))] => Float64(v))
	end
	return out
end

# A result may be a wrapper carrying `.info`, or `build_model`'s `meta` NamedTuple
# itself — a backend with no ROM wrapper passes the latter directly.
_info_of(x) = hasproperty(x, :info) ? getproperty(x, :info) : nothing
_info_of(x::NamedTuple) = hasproperty(x, :info) ? getproperty(x, :info) : x

"""
	write_summary(io, path, case; rom = nothing, title, metadata = Pair[],
				  append = true) -> Nothing

Print a run banner and timing table to `io`, and write `key: value` rows to
`path`.

The skeleton is shared: title, problem size, stage timings gathered by
[`stage_timings`](@ref) from both `case.info` and `rom.info`, then whatever
[`summary_entries`](@ref) the physics contributes, then `metadata` from the
caller (which wins over both — it is the most specific).

`append = true` is the default because `MORFE.save_rom` writes this same file
first, with the run's provenance (julia version, commit, timestamp). Opening it
with `"w"` here would silently discard that.
"""
function write_summary(io::IO, path::AbstractString, case;
	rom = nothing,
	title::AbstractString = "Run summary",
	metadata::AbstractVector = Pair{String, Any}[],
	append::Bool = true)
	ci, ri = _info_of(case), _info_of(rom)
	sep = "─"^78

	# ── Console banner ──────────────────────────────────────────────────────
	println(io)
	println(io, sep)
	println(io, title)
	if ci !== nothing && haskey(ci, :n_dofs)
		println(io, "  FOM size : $(ci.n_dofs) free DOFs")
	elseif ci !== nothing && haskey(ci, :n_free)
		println(io, "  FOM size : $(ci.n_free) free DOFs")
	end

	# A ROM's info commonly carries the case's info forward (SVK splats `m.info...`,
	# the fluid `build_model` splats the assembled model's), so the same stage can
	# appear on both sides. Keep the first occurrence: double-counting them would
	# also inflate the total.
	timings = Pair{String, Float64}[]
	for (label, t) in vcat(stage_timings(ci), stage_timings(ri))
		any(p -> first(p) == label, timings) || push!(timings, label => t)
	end
	if !isempty(timings)
		println(io, "─"^78)
		for (label, t) in timings
			@printf(io, "  %-32s  %9.3f s\n", label, t)
		end
		@printf(io, "  %-32s  %9.3f s\n", "TOTAL (timed stages)", sum(last, timings))
	end
	println(io, sep)

	# ── The file. Later rows override earlier ones, so the order is
	#    skeleton → physics → caller: most specific last. ────────────────────
	rows = Pair{String, Any}[]
	ci !== nothing && haskey(ci, :backend) && push!(rows, "backend" => ci.backend)
	if ci !== nothing
		haskey(ci, :n_dofs) && push!(rows, "n_free" => ci.n_dofs)
		haskey(ci, :n_free) && push!(rows, "n_free" => ci.n_free)
	end
	rom !== nothing && hasproperty(rom, :order) && push!(rows, "order" => rom.order)
	for (label, t) in timings
		push!(rows, "$(label)_time_s" => @sprintf("%.3f", t))
	end
	append!(rows, summary_entries(case, rom))
	append!(rows, metadata)

	seen = Dict{String, Any}()
	order = String[]
	for (k, v) in rows
		haskey(seen, k) || push!(order, k)
		seen[k] = v
	end
	open(path, append ? "a" : "w") do f
		for k in order
			println(f, "$k: $(seen[k])")
		end
	end
	return nothing
end
