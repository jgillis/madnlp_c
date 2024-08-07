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

# using MadNLP: MadNLPExecutionStats
using NLPModels

using MadNLPMumps
using MadNLPGPU
using CUDA
using UnsafePointers

AVAILABLE_SOLVERS = Set([
  "mumps",
  "umfpack",
  "lapack_cpu",
  "cudss",
  "cucholesky",
  "lapack_gpu"
])

export MadnlpCSolver, MadnlpCInterface, MadnlpCNumericIn, MadnlpCNumericOut, MadnlpCStats, madnlp_c_get_stats, madnlp_c_startup, madnlp_c_shutdown, madnlp_c_create, madnlp_c_option_type, madnlp_c_set_option_double, madnlp_c_set_option_bool, madnlp_c_set_option_int, madnlp_c_set_option_string, madnlp_c_input, madnlp_c_output, madnlp_c_solve, madnlp_c_destroy



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
  
    nw::Int64
    nc::Int64

    nzj_i::Ptr{Int64}
    nzj_j::Ptr{Int64}
    nzh_i::Ptr{Int64}
    nzh_j::Ptr{Int64}

    nnzj::Int64
    nnzh::Int64
    nnzo::Int64
  
    user_data::Ptr{Cvoid}
end

mutable struct MadnlpCNumericIn{T}
    x0::T
    l0::T
    lbx::T
    ubx::T
    lbg::T
    ubg::T
    # No-argument constructor
    MadnlpCNumericIn{T}() where T = new{T}()
    MadnlpCNumericIn{T}(x0,l0,lbx,ubx,lbg,ubg) where {T} = new(x0,l0,lbx,ubx,lbg,ubg)
end
MadnlpCNumericIn(x0::T,l0::T,lbx::T,ubx::T,lbg::T,ubg::T) where {T} = MadnlpCNumericIn{T}(x0,l0,lbx,ubx,lbg,ubg);

mutable struct MadnlpCNumericOut
  sol::Ptr{Cdouble}
  con::Ptr{Cdouble}
  obj::Ptr{Cdouble}
  mul::Ptr{Cdouble}
  mul_L::Ptr{Cdouble}
  mul_U::Ptr{Cdouble}
  MadnlpCNumericOut() = new()
end

mutable struct MadnlpCStats
  iter::Int64
  status::Int64
  dual_feas::Float64
  primal_feas::Float64
  MadnlpCStats() = new()
end

mutable struct MadnlpCSolver
  nlp_interface::MadnlpCInterface
  linear_solver_c::Ptr{Int8}
  linear_solver::String
  max_iter::Int64
  tol::Float64
  print_level::Int64
  minimize::Bool
  in_c::MadnlpCNumericIn{Ptr{Cdouble}}
  out_c::MadnlpCNumericOut
  stats_c::MadnlpCStats
  in::MadnlpCNumericIn{Vector{Float64}}

  nzj_i::Vector{Int64}
  nzj_j::Vector{Int64}
  nzh_i::Vector{Int64}
  nzh_j::Vector{Int64}

  solution::Vector{Float64}
  objective::Float64
  constraints::Vector{Float64}
  multipliers::Vector{Float64}
  multipliers_L::Vector{Float64}
  multipliers_U::Vector{Float64}

  MadnlpCSolver() = new()
end


ref_store::Dict{Ptr{MadnlpCSolver},Union{Ref{MadnlpCSolver},Nothing}} = Dict()



function NLPModels.jac_structure!(nlp::GenericModel, I::AbstractVector{T}, J::AbstractVector{T}) where T
  copyto!(I, nlp.nzj_i)
  copyto!(J, nlp.nzj_j)
end

function NLPModels.hess_structure!(nlp::GenericModel, I::AbstractVector{T}, J::AbstractVector{T}) where T
  copyto!(I, nlp.nzh_i)
  copyto!(J, nlp.nzh_j)
end

function NLPModels.obj(nlp::GenericModel, x::AbstractVector)
  Cx::Ptr{Cdouble} = Base.unsafe_convert(Ptr{Cdouble}, x)
  Cf::Ptr{Cdouble} = Base.unsafe_convert(Ptr{Cdouble}, nlp.obj)
  ret::Cint = ccall(nlp.eval_f, Cint, (Ptr{Cdouble},Ptr{Cdouble}, Ptr{Cvoid}), Cx, Cf, nlp.user_data)
  if Bool(ret)
     throw(Exception("CallbackError eval_f"))
  end
  return  nlp.obj[1]
end

function NLPModels.obj(nlp::GenericModel, x::CuArray)
  copyto!(nlp.bf.x, x)
  Cx::Ptr{Cdouble} = Base.unsafe_convert(Ptr{Cdouble}, nlp.bf.x)
  Cf::Ptr{Cdouble} = Base.unsafe_convert(Ptr{Cdouble}, nlp.obj)
  ret::Cint = ccall(nlp.eval_f, Cint, (Ptr{Cdouble},Ptr{Cdouble}, Ptr{Cvoid}), Cx, Cf, nlp.user_data)
  if Bool(ret)
     throw(Exception("CallbackError eval_f"))
  end
  return nlp.obj[1]
end

function NLPModels.cons!(nlp::GenericModel, x::AbstractVector, c::AbstractVector)
  Cx::Ptr{Cdouble} = Base.unsafe_convert(Ptr{Cdouble}, x)
  Cc::Ptr{Cdouble} = Base.unsafe_convert(Ptr{Cdouble}, c)
  ret::Cint = ccall(nlp.eval_g, Cint, (Ptr{Cdouble},Ptr{Cdouble},Ptr{Cvoid}), Cx, Cc, nlp.user_data)
  if Bool(ret)
     throw(Exception("CallbackError eval_cons"))
  end
  return c
end

function NLPModels.cons!(nlp::GenericModel, x::CuArray, c::CuArray)
  copyto!(nlp.bf.x, x)
  Cx::Ptr{Cdouble} = Base.unsafe_convert(Ptr{Cdouble}, nlp.bf.x)
  Cc::Ptr{Cdouble} = Base.unsafe_convert(Ptr{Cdouble}, nlp.bf.cons)
  ret::Cint = ccall(nlp.eval_g, Cint, (Ptr{Cdouble},Ptr{Cdouble},Ptr{Cvoid}), Cx, Cc, nlp.user_data)
  if Bool(ret)
     throw(Exception("CallbackError eval_cons"))
  end
  copyto!(c, nlp.bf.cons)
  return c
end

function NLPModels.grad!(nlp::GenericModel, x::AbstractVector, g::AbstractVector)
  Cx::Ptr{Cdouble} = Base.unsafe_convert(Ptr{Cdouble}, x)
  Cg::Ptr{Cdouble} = Base.unsafe_convert(Ptr{Cdouble}, g)
  ret::Cint = ccall(nlp.eval_grad_f, Cint, (Ptr{Cdouble},Ptr{Cdouble},Ptr{Cvoid}), Cx, Cg, nlp.user_data)
  if Bool(ret)
     throw(Exception("CallbackError eval_grad_f"))
  end
  # g = unsafe_wrap(Array, Cg, nlp.meta.nnzo)
  return g
end

function NLPModels.grad!(nlp::GenericModel, x::CuArray, g::CuArray)
  copyto!(nlp.bf.x, x)
  Cx::Ptr{Cdouble} = Base.unsafe_convert(Ptr{Cdouble}, nlp.bf.x)
  Cg::Ptr{Cdouble} = Base.unsafe_convert(Ptr{Cdouble}, nlp.bf.grad_f)
  ret::Cint = ccall(nlp.eval_grad_f, Cint, (Ptr{Cdouble},Ptr{Cdouble},Ptr{Cvoid}), Cx, Cg, nlp.user_data)
  if Bool(ret)
     throw(Exception("CallbackError eval_grad_f"))
  end
  copyto!(g, nlp.bf.grad_f)
  return g
end

function NLPModels.jac_coord!(nlp::GenericModel, x::AbstractVector, J::AbstractVector)
  Cx::Ptr{Cdouble} = Base.unsafe_convert(Ptr{Cdouble}, x)
  CJ::Ptr{Cdouble} = Base.unsafe_convert(Ptr{Cdouble}, J)
  ret::Cint = ccall(nlp.eval_jac_g, Cint, (Ptr{Cdouble},Ptr{Cdouble},Ptr{Cvoid}), Cx, CJ, nlp.user_data)
  if Bool(ret)
     throw(Exception("CallbackError eval_jac_g"))
  end
  J = unsafe_wrap(Array, CJ, nlp.meta.nnzj)
  return J
end

function NLPModels.jac_coord!(nlp::GenericModel, x::CuArray, J::CuArray)
  copyto!(nlp.bf.x, x)
  Cx::Ptr{Cdouble} = Base.unsafe_convert(Ptr{Cdouble}, nlp.bf.x)
  CJ::Ptr{Cdouble} = Base.unsafe_convert(Ptr{Cdouble}, nlp.bf.jac_g)
  ret::Cint = ccall(nlp.eval_jac_g, Cint, (Ptr{Cdouble},Ptr{Cdouble},Ptr{Cvoid}), Cx, CJ, nlp.user_data)
  if Bool(ret)
     throw(Exception("CallbackError eval_jac_g"))
  end
  copyto!(J, nlp.bf.jac_g)
  return J
end

function NLPModels.hess_coord!(nlp::GenericModel, x::AbstractVector, y::AbstractVector, H::AbstractVector;
                                obj_weight::Float64=1.0)
  Cx::Ptr{Cdouble} = Base.unsafe_convert(Ptr{Cdouble}, x)
  Cy::Ptr{Cdouble} = Base.unsafe_convert(Ptr{Cdouble}, y)
  CH::Ptr{Cdouble} = Base.unsafe_convert(Ptr{Cdouble}, H)
  ret::Cint = ccall(nlp.eval_h, Cint,
                    (Float64, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cvoid}),
                    obj_weight, Cx, Cy, CH, nlp.user_data)
  if Bool(ret)
     throw(Exception("CallbackError eval_hess_l"))
  end
  H = unsafe_wrap(Array, CH, nlp.meta.nnzh)
  return H
end

function NLPModels.hess_coord!(nlp::GenericModel, x::CuArray, y::CuArray, H::CuArray;
                               obj_weight::Cdouble=1.0)
  copyto!(nlp.bf.x, x)
  copyto!(nlp.bf.l, y)
  Cx::Ptr{Cdouble} = Base.unsafe_convert(Ptr{Cdouble}, nlp.bf.x)
  Cy::Ptr{Cdouble} = Base.unsafe_convert(Ptr{Cdouble}, nlp.bf.l)
  CH::Ptr{Cdouble} = Base.unsafe_convert(Ptr{Cdouble}, nlp.bf.hess_l)
  ret::Cint = ccall(nlp.eval_h, Cint,
                    (Float64, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cvoid}),
                    obj_weight, Cx, Cy, CH, nlp.user_data)
  if Bool(ret)
     throw(Exception("CallbackError eval_hess_l"))
  end
  copyto!(H, nlp.bf.hess_l)
  return H
end


function set_option(s::Ptr{MadnlpCSolver}, name::String, value::Any)
  s_jl::MadnlpCSolver = unsafe_pointer_to_objref(s)
  if name == "tol"
    if value <= 1e-15 value = 1e-15 end
    s_jl.tol = Float64(value)
  elseif name == "print_level"
    if value > 5 value = 5 end
    if value < 0 value = 0 end
    s_jl.print_level = Int(value)
  elseif name == "linear_solver"
    if !in(value, AVAILABLE_SOLVERS)
      @warn "Linear solver $value not supported, available solver are:\n$AVAILABLE_SOLVERS"
    else
      s_jl.linear_solver = value
    end
  elseif name == "max_iter"
    if value < 0 value = 0 end
    s_jl.max_iter = Int(value)
  elseif name == "minimize"
    s_jl.minimize = Bool(value)
  # julia runtime options
  elseif name == "trace_compile"
  elseif name == "compile"
  elseif name == "trace_compile_output"
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
  # Create the solver object
  solver = MadnlpCSolver()
  solver.nlp_interface = unsafe_load(nlp_interface)
  solver.tol = 1e-8
  solver.linear_solver = "mumps"
  solver.linear_solver_c = Base.unsafe_convert(Ptr{Int8}, solver.linear_solver)
  solver.max_iter = 3000
  solver.print_level = 3
  solver.minimize = true

  interf = solver.nlp_interface

  solver.in = MadnlpCNumericIn{Vector{Float64}}()

  solver.in.x0 = fill(0.0, interf.nw)
  solver.in.l0 = fill(0.0, interf.nc)
  solver.in.lbx = fill(-Inf, interf.nw)
  solver.in.ubx = fill(Inf, interf.nw)
  solver.in.lbg = fill(0.0, interf.nc)
  solver.in.ubg = fill(0.0, interf.nc)

  solver.in_c = MadnlpCNumericIn{Ptr{Cdouble}}()
  solver.in_c.x0 = Base.unsafe_convert(Ptr{Cdouble},  solver.in.x0)
  solver.in_c.l0 = Base.unsafe_convert(Ptr{Cdouble},  solver.in.l0)
  solver.in_c.lbx = Base.unsafe_convert(Ptr{Cdouble}, solver.in.lbx)
  solver.in_c.ubx = Base.unsafe_convert(Ptr{Cdouble}, solver.in.ubx)
  solver.in_c.lbg = Base.unsafe_convert(Ptr{Cdouble}, solver.in.lbg)
  solver.in_c.ubg = Base.unsafe_convert(Ptr{Cdouble}, solver.in.ubg)

  solver.out_c = MadnlpCNumericOut()
  solver.stats_c = MadnlpCStats()

  # nzj_i may not be properly aligned, so copying is neededs
  solver.nzj_i = Vector{Int64}(undef, interf.nnzj)
  solver.nzj_j = Vector{Int64}(undef, interf.nnzj)
  solver.nzh_i = Vector{Int64}(undef, interf.nnzh)
  solver.nzh_j = Vector{Int64}(undef, interf.nnzh)

  unsafe_copyto!(Base.unsafe_convert(Ptr{Int64}, solver.nzj_i), interf.nzj_i, interf.nnzj)
  unsafe_copyto!(Base.unsafe_convert(Ptr{Int64}, solver.nzj_j), interf.nzj_j, interf.nnzj)
  unsafe_copyto!(Base.unsafe_convert(Ptr{Int64}, solver.nzh_i), interf.nzh_i, interf.nnzh)
  unsafe_copyto!(Base.unsafe_convert(Ptr{Int64}, solver.nzh_j), interf.nzh_j, interf.nnzh)

  solver.solution = Vector{Float64}(undef, interf.nw)
  solver.objective = 0.0
  solver.constraints = Vector{Float64}(undef, interf.nc)
  solver.multipliers = Vector{Float64}(undef, interf.nc)
  solver.multipliers_L = Vector{Float64}(undef, interf.nw)
  solver.multipliers_U = Vector{Float64}(undef, interf.nw)

  solver_ref = Ref(solver)

  solver_ptr = Base.unsafe_convert(Ptr{MadnlpCSolver}, solver_ref)

  ref_store[solver_ptr] = solver_ref

  # Return the pointer to the solver object
  return solver_ptr
end

Base.@ccallable function madnlp_c_input(s::Ptr{MadnlpCSolver})::Ptr{MadnlpCNumericIn{Ptr{Cdouble}}}
  solver = unsafe_pointer_to_objref(s)

  return pointer_from_objref(solver.in_c)
end

Base.@ccallable function madnlp_c_output(s::Ptr{MadnlpCSolver})::Ptr{MadnlpCNumericOut}
  solver = unsafe_pointer_to_objref(s)

  return pointer_from_objref(solver.out_c)
end

Base.@ccallable function madnlp_c_get_stats(s::Ptr{MadnlpCSolver})::Ptr{MadnlpCStats}
  solver = unsafe_pointer_to_objref(s)

  return pointer_from_objref(solver.stats_c)
end

Base.@ccallable function madnlp_c_option_type(name::Ptr{Cchar})::Cint
  n = unsafe_string(name)
  if n == "tol" return 0 end
  if n == "max_iter" return 1 end
  if n == "print_level" return 1 end
  if n == "minimize" return 2 end
  if n == "trace_compile" return 2 end
  if n == "compile" return 3 end
  if n == "linear_solver" return 3 end
  if n == "trace_compile_output" return 3 end
  return -1
end

Base.@ccallable function madnlp_c_set_option_double(s::Ptr{MadnlpCSolver}, name::Ptr{Cchar}, val::Cdouble)::Cint
  try
    set_option(s, unsafe_string(name), val)
  catch e
    @warn e
    return 1
  end
  return 0
end

Base.@ccallable function madnlp_c_set_option_bool(s::Ptr{MadnlpCSolver}, name::Ptr{Cchar}, val::Int64)::Cint
  try
    set_option(s, unsafe_string(name), val)
  catch e
    @warn e
    return 1
  end
  return 0
end

Base.@ccallable function madnlp_c_set_option_int(s::Ptr{MadnlpCSolver}, name::Ptr{Cchar}, val::Int64)::Cint
  try
    set_option(s, unsafe_string(name), val)
  catch e
    @warn e
    return 1
  end
  return 0
end

Base.@ccallable function madnlp_c_set_option_string(s::Ptr{MadnlpCSolver}, name::Ptr{Cchar}, val::Ptr{Cchar})::Cint
  try
    set_option(s, unsafe_string(name), unsafe_string(val))
  catch e
    @warn e
    return 1
  end
  return 0
end


Base.@ccallable function madnlp_c_solve(s::Ptr{MadnlpCSolver})::Cint

  solver = unsafe_pointer_to_objref(s)
  nlp_interface = solver.nlp_interface
  nvar = nlp_interface.nw
  ncon = nlp_interface.nc

  in_c = solver.in_c

  x0 = unsafe_wrap(Array, in_c.x0, (nlp_interface.nw,))
  y0 = unsafe_wrap(Array, in_c.l0, (nlp_interface.nc,))
  lvar = unsafe_wrap(Array, in_c.lbx, (nlp_interface.nw,))
  uvar = unsafe_wrap(Array, in_c.ubx, (nlp_interface.nw,))
  lcon = unsafe_wrap(Array, in_c.lbg, (nlp_interface.nc,))
  ucon = unsafe_wrap(Array, in_c.ubg, (nlp_interface.nc,))

  # default values
  main_log_level = Logging.Warn
  madnlp_log = MadNLP.NOTICE

  if solver.print_level == 0
    main_log_level = Logging.Error
    madnlp_log = MadNLP.ERROR
  elseif solver.print_level == 1
    main_log_level = Logging.Warn
    madnlp_log = MadNLP.WARN
  elseif solver.print_level == 2
    main_log_level = Logging.Warn
    madnlp_log = MadNLP.NOTICE
  elseif solver.print_level == 3
    main_log_level = Logging.Info
    madnlp_log = MadNLP.INFO
  elseif solver.print_level == 4
    main_log_level = Logging.Debug
    madnlp_log = MadNLP.DEBUG
  elseif solver.print_level == 5
    main_log_level = Logging.Debug
    madnlp_log = MadNLP.TRACE
  end

  logger = ConsoleLogger(stderr, main_log_level)
  global_logger(logger)

  GPU_DEVICE::Bool = false

  nzj_i = solver.nzj_i
  nzj_j = solver.nzj_j
  nzh_i = solver.nzh_i
  nzh_j = solver.nzh_j

  linear_solver = nothing
  if solver.linear_solver == "mumps"
    linear_solver = MadNLPMumps.MumpsSolver
  elseif solver.linear_solver == "umfpack"
    linear_solver = UmfpackSolver
  elseif solver.linear_solver == "lapack_cpu"
    linear_solver = LapackCPUSolver
  elseif solver.linear_solver == "cudss"
    linear_solver = MadNLPGPU.CUDSSSolver
    GPU_DEVICE = true
  elseif solver.linear_solver == "lapack_gpu"
    linear_solver = MadNLPGPU.LapackGPUSolver
    GPU_DEVICE = true
  elseif solver.linear_solver == "cucholesky"
    linear_solver = MadNLPGPU.CuCholeskySolver
    GPU_DEVICE = true
  else
    @warn "$(solver.linear_solver) not available"
  end

  buffers = nothing
  if GPU_DEVICE
    Gx0 = CuArray{Float64}(undef, nvar)
    Gy0 = CuArray{Float64}(undef, ncon)
    Glvar = CuArray{Float64}(undef, nvar)
    Guvar = CuArray{Float64}(undef, nvar)
    Glcon = CuArray{Float64}(undef, ncon)
    Gucon = CuArray{Float64}(undef, ncon)
    Gnzj_i = CuArray{Int64}(undef, nlp_interface.nnzj)
    Gnzj_j = CuArray{Int64}(undef, nlp_interface.nnzj)
    Gnzh_i = CuArray{Int64}(undef, nlp_interface.nnzh)
    Gnzh_j = CuArray{Int64}(undef, nlp_interface.nnzh)

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

  meta = NLPModelMeta(
    nvar,
    ncon = ncon,
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
  )

  nlp = GenericModel(
    meta,
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

  # res::Union{MadNLPExecutionStats{Float64, CuArray{Float64}},MadNLPExecutionStats{Float64, Vector{Float64}},Nothing} = nothing;
  # res::Union{MadNLPExecutionStats{Float64, CuArray{Float64}},MadNLPExecutionStats{Float64, Vector{Float64}},Nothing} = nothing;

  local res
  try
    res = MadNLP.solve!(madnlp_solver, max_iter = Int(solver.max_iter), tol=solver.tol)
  catch e
    @error e
    return 1
  end

  solver.stats_c.iter = res.iter
  solver.stats_c.status = Integer(res.status)
  solver.stats_c.dual_feas = res.dual_feas
  solver.stats_c.primal_feas = res.primal_feas

  copyto!(solver.solution, res.solution)
  solver.objective = res.objective
  copyto!(solver.constraints, res.constraints)
  copyto!(solver.multipliers, res.multipliers)
  copyto!(solver.multipliers_L, res.multipliers_L)
  copyto!(solver.multipliers_U, res.multipliers_U)

  # Make results available to C
  solver.out_c.sol = Base.unsafe_convert(Ptr{Cdouble},solver.solution)
  solver.out_c.obj = Base.unsafe_convert(Ptr{Cdouble},[solver.objective])
  solver.out_c.con = Base.unsafe_convert(Ptr{Cdouble},solver.constraints)
  solver.out_c.mul = Base.unsafe_convert(Ptr{Cdouble},solver.multipliers)
  solver.out_c.mul_L = Base.unsafe_convert(Ptr{Cdouble},solver.multipliers_L)
  solver.out_c.mul_U = Base.unsafe_convert(Ptr{Cdouble},solver.multipliers_U)

  return 0

end


Base.@ccallable function madnlp_c_destroy(s::Ptr{MadnlpCSolver})::Cvoid
  ref_store[s] = nothing
  #GC.gc()
end

end # module MadNLP_C
