module MadNLP_C
"""
Solve Generic formulated as the NLP:

minimize     f(x1,x2,..,xN)
subject to   g1(x1,...,xN) == 0
subject to   ...
subject to   g2(x1,...,xN) == 0
"""

using Logging
using Base

# using MadNLP
using NLPModels

using MadNLPMumps
using MadNLPGPU
using CUDA
using UnsafePointers

export MadnlpCSolver, MadnlpCInterface, MadnlpCNumericIn, MadnlpCNumericOut, madnlp_c_get_stats, madnlp_c_startup, madnlp_c_shutdown, madnlp_c_create, madnlp_c_option_type, madnlp_c_set_option_double, madnlp_c_set_option_bool, madnlp_c_set_option_int, madnlp_c_set_option_string, madnlp_c_solve, madnlp_c_destroy

struct CPUBuffers
	x::Vector{Float64}
	grad_f::Vector{Float64}
	l::Vector{Float64}
	cons::Vector{Float64}
	jac_g::Vector{Float64}
	hess_l::Vector{Float64}
end

struct GenericModel{T, VT, IVT, FT} <: AbstractNLPModel{T,VT}
	meta::NLPModelMeta{T, VT}
	counters::NLPModels.Counters
	bf::Union{CPUBuffers,Nothing}
	obj::Vector{Float64}
	nzj_i::IVT
	nzj_j::IVT
	nzh_i::IVT
	nzh_j::IVT
	eval_f::FT
	eval_g::FT
	eval_grad_f::FT
	eval_jac_g::FT
	eval_h::FT
	user_data::FT
end

mutable struct MadnlpCInterface
    eval_obj::Ptr{Cvoid}
    eval_constr::Ptr{Cvoid}
    eval_obj_grad::Ptr{Cvoid}
    eval_constr_jac::Ptr{Cvoid}
    eval_lag_hess::Ptr{Cvoid}

    nw::Csize_t
    nc::Csize_t

    nzj_i::Ptr{Csize_t}
    nzj_j::Ptr{Csize_t}
    nzh_i::Ptr{Csize_t}
    nzh_j::Ptr{Csize_t}

    nnzj::Csize_t
    nnzh::Csize_t
    nnzo::Csize_t

    user_data::Ptr{Cvoid}
end

mutable struct MadnlpCSolver
    nlp_interface::MadnlpCInterface
    lin_solver_id::Int64
    max_iters::Int64
    print_level::Int64
    minimize::Bool
    res::MadNLP.MadNLPExecutionStats{Float64, Vector{Float64}}
    MadnlpCSolver() = new()
end

struct MadnlpCStats
    iter::Int64
end

mutable struct MadnlpCNumericIn
    x0::Ptr{Cdouble}
    l0::Ptr{Cdouble}
    lbx::Ptr{Cdouble}
    ubx::Ptr{Cdouble}
    lbg::Ptr{Cdouble}
    ubg::Ptr{Cdouble}
end

mutable struct MadnlpCNumericOut
    sol::Ptr{Cdouble}
    con::Ptr{Cdouble}
    obj::Ptr{Cdouble}
    mul::Ptr{Cdouble}
    mul_L::Ptr{Cdouble}
    mul_U::Ptr{Cdouble}
    MadnlpCNumericOut() = new()
end

function NLPModels.jac_structure!(nlp::GenericModel, I::AbstractVector{T}, J::AbstractVector{T}) where T
	@info "jac_strc"
	@info "nzj_i" nlp.nzj_i
	@info "nzj_j" nlp.nzj_j
	copyto!(I, nlp.nzj_i)
	copyto!(J, nlp.nzj_j)
	@info "I"   I
	@info "J"   J
end

function NLPModels.hess_structure!(nlp::GenericModel, I::AbstractVector{T}, J::AbstractVector{T}) where T
	@info "hess_strc"
	@info "nzh_i" nlp.nzh_i
	@info "nzh_j" nlp.nzh_j
	copyto!(I, nlp.nzh_i)
	copyto!(J, nlp.nzh_j)
	@info "I"   I
	@info "J"   J
end

function NLPModels.obj(nlp::GenericModel, x::AbstractVector)
	@info "obj x=" x
	Cx::Ptr{Cdouble} = Base.unsafe_convert(Ptr{Cdouble}, x)
	Cf::Ptr{Cdouble} = Base.unsafe_convert(Ptr{Cdouble}, nlp.obj)
	ret::Cint = ccall(nlp.eval_f, Cint, (Ptr{Cdouble},Ptr{Cdouble}, Ptr{Cvoid}), Cx, Cf, nlp.user_data)
	if Bool(ret) @error "function call failed" end
	#WARNING unsafe_wrap pointer must be 8-aligned!
	@info "obj = " nlp.obj
	return  nlp.obj[1]
end

function NLPModels.obj(nlp::GenericModel, x::CuArray)
	copyto!(nlp.bf.x, x)
	Cx::Ptr{Cdouble} = Base.unsafe_convert(Ptr{Cdouble}, nlp.bf.x)
	# f::Vector{Float64} = Vector{Float64}(undef,1)
	@info "obj-in" nlp.bf.x
	Cf::Ptr{Cdouble} = Base.unsafe_convert(Ptr{Cdouble}, nlp.obj)
	ret::Cint = ccall(nlp.eval_f, Cint, (Ptr{Cdouble},Ptr{Cdouble}, Ptr{Cvoid}), Cx, Cf, nlp.user_data)
	if Bool(ret) @error "function call failed" end
	@info "obj-out" nlp.obj
	return nlp.obj[1]
end

function NLPModels.cons!(nlp::GenericModel, x::AbstractVector, c::AbstractVector)
	@info "cons x = " x
	Cx::Ptr{Cdouble} = Base.unsafe_convert(Ptr{Cdouble}, x)
	Cc::Ptr{Cdouble} = Base.unsafe_convert(Ptr{Cdouble}, c)
	ret::Cint = ccall(nlp.eval_g, Cint, (Ptr{Cdouble},Ptr{Cdouble},Ptr{Cvoid}), Cx, Cc, nlp.user_data)
	if Bool(ret) @error "function call failed" end
	@info "cons = " c
	return c
end

function NLPModels.cons!(nlp::GenericModel, x::CuArray, c::CuArray)
	copyto!(nlp.bf.x, x)
	@info "GPU cons in" nlp.bf.x
	Cx::Ptr{Cdouble} = Base.unsafe_convert(Ptr{Cdouble}, nlp.bf.x)
	Cc::Ptr{Cdouble} = Base.unsafe_convert(Ptr{Cdouble}, nlp.bf.cons)
	ret::Cint = ccall(nlp.eval_g, Cint, (Ptr{Cdouble},Ptr{Cdouble},Ptr{Cvoid}), Cx, Cc, nlp.user_data)
	if Bool(ret) @error "GPU cons failed" end
	@info "GPU cons out" nlp.bf.cons
	copyto!(c, nlp.bf.cons)
	return c
end

function NLPModels.grad!(nlp::GenericModel, x::AbstractVector, g::AbstractVector)
	@info "grad-in"
	Cx::Ptr{Cdouble} = Base.unsafe_convert(Ptr{Cdouble}, x)
	Cg::Ptr{Cdouble} = Base.unsafe_convert(Ptr{Cdouble}, g)
	@info "grad-in" x
	ret::Cint = ccall(nlp.eval_grad_f, Cint, (Ptr{Cdouble},Ptr{Cdouble},Ptr{Cvoid}), Cx, Cg, nlp.user_data)
	if Bool(ret) @error "function call failed" end
	@info "grad-out" g
	# g = unsafe_wrap(Array, Cg, nlp.meta.nnzo)
	return g
end

function NLPModels.grad!(nlp::GenericModel, x::CuArray, g::CuArray)
	copyto!(nlp.bf.x, x)
	@info "GPU grad in" nlp.bf.x
	Cx::Ptr{Cdouble} = Base.unsafe_convert(Ptr{Cdouble}, nlp.bf.x)
	Cg::Ptr{Cdouble} = Base.unsafe_convert(Ptr{Cdouble}, nlp.bf.grad_f)
	ret::Cint = ccall(nlp.eval_grad_f, Cint, (Ptr{Cdouble},Ptr{Cdouble},Ptr{Cvoid}), Cx, Cg, nlp.user_data)
	if Bool(ret) @error "GPU grad failed" end
	@info "GPU grad out" nlp.bf.grad_f
	copyto!(g, nlp.bf.grad_f)
	return g
end

function NLPModels.jac_coord!(nlp::GenericModel, x::AbstractVector, J::AbstractVector)
	@info "jac_g"
	Cx::Ptr{Cdouble} = Base.unsafe_convert(Ptr{Cdouble}, x)
	CJ::Ptr{Cdouble} = Base.unsafe_convert(Ptr{Cdouble}, J)
	ret::Cint = ccall(nlp.eval_jac_g, Cint, (Ptr{Cdouble},Ptr{Cdouble},Ptr{Cvoid}), Cx, CJ, nlp.user_data)
	if Bool(ret) @error "function call failed" end
  J = unsafe_wrap(Array, CJ, nlp.meta.nnzj)
	@info "jac_g"
	return J
end

function NLPModels.jac_coord!(nlp::GenericModel, x::CuArray, J::CuArray)
	copyto!(nlp.bf.x, x)
	@info "GPU jac in" nlp.bf.x
	# copyto!(nlp.bf.jac_g, J)
	Cx::Ptr{Cdouble} = Base.unsafe_convert(Ptr{Cdouble}, nlp.bf.x)
	CJ::Ptr{Cdouble} = Base.unsafe_convert(Ptr{Cdouble}, nlp.bf.jac_g)
	ret::Cint = ccall(nlp.eval_jac_g, Cint, (Ptr{Cdouble},Ptr{Cdouble},Ptr{Cvoid}), Cx, CJ, nlp.user_data)
	if Bool(ret) @error "GPU jac failed" end
	# J = unsafe_wrap(Array, CJ, nlp.meta.nnzj)
	@info "GPU jac out" nlp.bf.jac_g
	copyto!(J, nlp.bf.jac_g)
	return J
end

function NLPModels.hess_coord!(nlp::GenericModel, x::AbstractVector, y::AbstractVector, H::AbstractVector;
                                obj_weight::Float64=1.0)
	@info "hess"
	Cx::Ptr{Cdouble} = Base.unsafe_convert(Ptr{Cdouble}, x)
	Cy::Ptr{Cdouble} = Base.unsafe_convert(Ptr{Cdouble}, y)
	CH::Ptr{Cdouble} = Base.unsafe_convert(Ptr{Cdouble}, H)
	ret::Cint = ccall(nlp.eval_h, Cint,
                    (Float64, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cvoid}),
                    obj_weight, Cx, Cy, CH, nlp.user_data)
	if Bool(ret) @error "function call failed" end
  H = unsafe_wrap(Array, CH, nlp.meta.nnzh)
	@info "hess" x H
	return H
end

function NLPModels.hess_coord!(nlp::GenericModel, x::CuArray, y::CuArray, H::CuArray;
                               obj_weight::Cdouble=1.0)
	copyto!(nlp.bf.x, x)
	copyto!(nlp.bf.l, y)
	# copyto!(nlp.bf.hess_l, H)
	@info "GPU hess x.in" nlp.bf.x
	@info "GPU hess l.in" nlp.bf.l
	Cx::Ptr{Cdouble} = Base.unsafe_convert(Ptr{Cdouble}, nlp.bf.x)
	Cy::Ptr{Cdouble} = Base.unsafe_convert(Ptr{Cdouble}, nlp.bf.l)
	CH::Ptr{Cdouble} = Base.unsafe_convert(Ptr{Cdouble}, nlp.bf.hess_l)
	ret::Cint = ccall(nlp.eval_h, Cint,
                    (Float64, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cvoid}),
                    obj_weight, Cx, Cy, CH, nlp.user_data)
	if Bool(ret) @error "GPU hess failed" end
	@info "GPU hess out" nlp.bf.grad_f
	copyto!(H, nlp.bf.hess_l)
  # H = unsafe_wrap(Array, CH, nlp.meta.nnzh)
	return H
end


function set_option(s::Ptr{MadnlpCSolver}, name::String, value::Any)
    if name == "print_level"
        s.print_level = Int(value)
        if s.print_level > 5 s.print_level = 5 end
        if s.print_level < 0 s.print_level = 0 end
    elseif name == "lin_solver_id"
        s.lin_solver_id = Int(value)
        if s.lin_solver_id > 5 s.lin_solver_id = 5 end
        if s.lin_solver_id < 0 s.lin_solver_id = 0 end
    else
        @warn "Unknown option $name"
    end
end

Base.@ccallable function madnlp_c_startup(argc::Cint, argv::Ptr{Ptr{Cchar}})::Cvoid
    init_julia(argc, argv)
end

Base.@ccallable function madnlp_c_shutdown()::Cvoid
    shutdown_julia(1)
end

Base.@ccallable function madnlp_c_create(nlp_interface::Ptr{MadnlpCInterface})::Ptr{MadnlpCSolver}
    # Allocate memory for the solver
    solver_ptr = Ptr{MadnlpCSolver}(Libc.malloc(sizeof(MadnlpCSolver)))

    # Create the solver object
    solver = MadnlpCSolver()
    solver.nlp_interface = unsafe_load(nlp_interface)
    solver.lin_solver_id = 0
    solver.max_iters = 1000
    solver.print_level = 5
    solver.minimize = true

    # Copy the solver object to the allocated memory
    unsafe_store!(solver_ptr, solver)

    # Return the pointer to the solver object
    return solver_ptr
end

Base.@ccallable function madnlp_c_option_type(name::Ptr{Cchar})::Cint
    n = unsafe_string(name)
    if n == "acceptable_tol" return 0 end
    if n == "bound_frac" return 0 end
    if n == "bound_push" return 0 end
    if n == "bound_relax_factor" return 0 end
    if n == "constr_viol_tol" return 0 end
    if n == "lammax" return 0 end
    if n == "mu_init" return 0 end
    if n == "recalc_y_feas_tol" return 0 end
    if n == "tol" return 0 end
    if n == "warm_start_mult_bound_push" return 0 end
    if n == "acceptable_iter" return 1 end
    if n == "max_iter" return 1 end
    if n == "print_level" return 1 end
    if n == "lin_solver_id" return 1 end
    if n == "iterative_refinement" return 2 end
    if n == "ls_scaling" return 2 end
    if n == "recalc_y" return 2 end
    if n == "warm_start_init_point" return 2 end
    return -1
end

Base.@ccallable function madnlp_c_set_option_double(s::Ptr{MadnlpCSolver}, name::Ptr{Cchar}, val::Cdouble)::Cint
    try
        set_option(s, unsafe_string(name), val)
    catch e
        return 1
    end
    return 0
end

Base.@ccallable function madnlp_c_set_option_bool(s::Ptr{MadnlpCSolver}, name::Ptr{Cchar}, val::Cint)::Cint
    try
        set_option(s, unsafe_string(name), Bool(val))
    catch e
        return 1
    end
    return 0
end

Base.@ccallable function madnlp_c_set_option_int(s::Ptr{MadnlpCSolver}, name::Ptr{Cchar}, val::Cint)::Cint
    try
        set_option(s, unsafe_string(name), val)
    catch e
        return 1
    end
    return 0
end

Base.@ccallable function madnlp_c_set_option_string(s::Ptr{MadnlpCSolver}, name::Ptr{Cchar}, val::Ptr{Cchar})::Cint
    try
        set_option(s, unsafe_string(name), unsafe_string(val))
    catch e
        return 1
    end
    return 0
end

Base.@ccallable function madnlp_c_get_stats(s::Ptr{MadnlpCSolver})::Ptr{MadnlpCStats}
    return 0
end

Base.@ccallable function madnlp_c_solve(s::Ptr{MadnlpCSolver},in::Ptr{MadnlpCNumericIn},out::Ptr{MadnlpCNumericOut})::Cint

    solver = unsafe_load(s)
    nlp_interface = solver.nlp_interface
    nvar = nlp_interface.nw
    ncon = nlp_interface.nc

    out_backup = out

    in = unsafe_load(in)
    out = UnsafePtr(out)

    # default values
    main_log_level = Logging.Warn
    madnlp_log = MadNLP.NOTICE

    if solver.print_level == 1
        main_log_level = Logging.Error
        madnlp_log = MadNLP.ERROR
    elseif solver.print_level == 2
        main_log_level = Logging.Warn
        madnlp_log = MadNLP.WARN
    elseif solver.print_level == 3
        main_log_level = Logging.Warn
        madnlp_log = MadNLP.NOTICE
    elseif solver.print_level == 4
        main_log_level = Logging.Info
        madnlp_log = MadNLP.INFO
    elseif solver.print_level == 5
        main_log_level = Logging.Debug
        madnlp_log = MadNLP.DEBUG
    elseif solver.print_level == 6
        main_log_level = Logging.Debug
        madnlp_log = MadNLP.TRACE
    end

    logger = ConsoleLogger(stderr, main_log_level)
    global_logger(logger)

    GPU_DEVICE::Bool = false

    stuff = in.lbx

    @info "in lbx" stuff

    x0 = unsafe_wrap(Array, in.x0, (nvar,))
    y0 = unsafe_wrap(Array, in.l0, (ncon,))
    lvar = unsafe_wrap(Array, in.lbx, (nvar,))
    uvar = unsafe_wrap(Array, in.ubx, (nvar,))
    lcon = unsafe_wrap(Array, in.lbg, (ncon,))
    ucon = unsafe_wrap(Array, in.ubg, (ncon,))

    nzj_i = unsafe_wrap(Array, nlp_interface.nzj_i, (nlp_interface.nnzj,))
    nzj_j = unsafe_wrap(Array, nlp_interface.nzj_j, (nlp_interface.nnzj,))
    nzh_i = unsafe_wrap(Array, nlp_interface.nzh_i, (nlp_interface.nnzh,))
    nzh_j = unsafe_wrap(Array, nlp_interface.nzh_j, (nlp_interface.nnzh,))

    @info "x0 julia" x0
    @info "lvar julia" lvar

    lin_solver_name = "none"
    if solver.lin_solver_id == 0
        lin_solver_name = "mumps"
        linear_solver = MadNLPMumps.MumpsSolver
    elseif solver.lin_solver_id == 1
        lin_solver_name = "umfpack"
        linear_solver = UmfpackSolver
    elseif solver.lin_solver_id == 2
        lin_solver_name = "lapackCPU"
        linear_solver = LapackCPUSolver
    elseif solver.lin_solver_id == 3
        lin_solver_name = "CUDSSS"
        linear_solver = MadNLPGPU.CUDSSSolver
        GPU_DEVICE = true
    elseif solver.lin_solver_id == 4
        lin_solver_name = "lapackGPU"
        linear_solver = MadNLPGPU.LapackGPUSolver
        GPU_DEVICE = true
    elseif solver.lin_solver_id == 5
        lin_solver_name = "CuCholesky"
        linear_solver = MadNLPGPU.CuCholeskySolver
        GPU_DEVICE = true
    end
    @info "Using linear solver $(lin_solver_name)"


    buffers = nothing
    if GPU_DEVICE
        Gx0 = CuArray{Float64}(undef, nvar)
        Gy0 = CuArray{Float64}(undef, ncon)
        Glvar = CuArray{Float64}(undef, nvar)
        Guvar = CuArray{Float64}(undef, nvar)
        Glcon = CuArray{Float64}(undef, ncon)
        Gucon = CuArray{Float64}(undef, ncon)
        Gnzj_i = CuArray{UInt64}(undef, nnzj)
        Gnzj_j = CuArray{UInt64}(undef, nnzj)
        Gnzh_i = CuArray{UInt64}(undef, nnzh)
        Gnzh_j = CuArray{UInt64}(undef, nnzh)

        copyto!(Gx0, x0)
        copyto!(Gy0, y0)
        copyto!(Glvar, lvar)
        copyto!(Guvar, uvar)
        copyto!(Glcon, lcon)
        copyto!(Gucon, ucon)
        copyto!(Gnzj_i, nzj_i)
        copyto!(Gnzj_j, nzj_j)
        copyto!(Gnzh_i, nzh_i)
        copyto!(Gnzh_j, nzh_j)

        x0 = Gx0
        y0 = Gy0
        lvar = Glvar
        uvar = Guvar
        lcon = Glcon
        ucon = Gucon
        nzj_i = Gnzj_i
        nzj_j = Gnzj_j
        nzh_i = Gnzh_i
        nzh_j = Gnzh_j

        buffers = CPUBuffers(
            Vector{Float64}(undef, nvar),
            Vector{Float64}(undef, nlp_interface.nnzo),
            Vector{Float64}(undef, ncon),
            Vector{Float64}(undef, ncon),
            Vector{Float64}(undef, nlp_interface.nnzj),
            Vector{Float64}(undef, nlp_interface.nnzh)
        )
    end

    @info "x0" x0
    @info "y0" y0
    @info "lvar" lvar
    @info "uvar" uvar
    @info "lcon" lcon
    @info "ucon" ucon

    nlp = GenericModel(
        NLPModelMeta(
            Int64(nvar),
            ncon = Int64(ncon),
            nnzo = nlp_interface.nnzo,
            nnzj = nlp_interface.nnzj,
            nnzh = nlp_interface.nnzh,
            x0 = x0,
            y0 = y0,
            lvar = lvar,
            uvar = uvar,
            lcon = lcon,
            ucon = ucon,
            name = "Generic",
            minimize = solver.minimize
        ),
        Counters(),
        buffers,
        Vector{Float64}(undef, 1),
        nzj_i,
        nzj_j,
        nzh_i,
        nzh_j,
        nlp_interface.eval_obj,
        nlp_interface.eval_constr,
        nlp_interface.eval_obj_grad,
        nlp_interface.eval_constr_jac,
        nlp_interface.eval_lag_hess,
        nlp_interface.user_data
    )

    madnlp_solver = MadNLPSolver(nlp; print_level = madnlp_log, linear_solver = linear_solver)
    solver.res = MadNLP.solve!(madnlp_solver, max_iter = Int(solver.max_iters))

    # Make results available to C
    out.sol[] = Base.unsafe_convert(Ptr{Cdouble},solver.res.solution)
    out.obj[] = Base.unsafe_convert(Ptr{Cdouble},Ref(solver.res.objective))
    out.con[] = Base.unsafe_convert(Ptr{Cdouble},solver.res.constraints)
    out.mul[] = Base.unsafe_convert(Ptr{Cdouble},solver.res.multipliers)
    out.mul_L[] = Base.unsafe_convert(Ptr{Cdouble},solver.res.multipliers_L)
    out.mul_U[] = Base.unsafe_convert(Ptr{Cdouble},solver.res.multipliers_U)

    return 0

end


Base.@ccallable function madnlp_c_destroy(s::Ptr{MadnlpCSolver})::Cvoid
    # Free the allocated memory
    Libc.free(s)
end

end # module MadNLP_C

