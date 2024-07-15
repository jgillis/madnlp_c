using MadNLP_C
using Base: unsafe_convert
using Logging

logger = ConsoleLogger(stderr, Logging.Warn)
global_logger(logger)


const nvar::Int64 = 2
const ncon::Int64 = 1

a::Float64 = 1.0
b::Float64 = 100.0

function _eval_f!(w::Vector{T}, f::Vector{T}) where T
	f[1] = (a-w[1])^2 + b*(w[2]-w[1]^2)^2
end

function _eval_g!(w::Vector{T}, g::Vector{T}) where T
	g[1] = w[1]^2 + w[2]^2 - 1
end

function _eval_jac_g!(w::Vector{T}, jac_g::Vector{T}) where T
	jac_g[1] = 2*w[1]
	jac_g[2] = 2*w[2]
end

function _eval_grad_f!(w::Vector{T}, g::Vector{T}) where T
	g[1] = -4*b*w[1]*(w[2]-w[1]^2)-2*(a-w[1])
	g[2] = b*2*(w[2]-w[1]^2)
end

function _eval_h!(w::Vector{T},l::Vector{T}, h::Vector{T}) where T
	h[1] = (+2 -4*b*w[2] +12*b*w[1]^2)
	h[2] = (-4*b*w[1])
	h[3] = (2*b)
end

function eval_f(Cw::Ptr{Cdouble},Cf::Ptr{Cdouble}, d::Ptr{Cvoid})::Cint
  w::Vector{Float64} = unsafe_wrap(Array, Cw, nvar)
  f::Vector{Float64} = unsafe_wrap(Array, Cf, 1)
  _eval_f!(w,f)
	return 0
end

function eval_g(w::Ptr{Cdouble},Ccons::Ptr{Cdouble}, d::Ptr{Cvoid})::Cint
	w::Vector{Float64} = unsafe_wrap(Array, w, nvar)
	cons::Vector{Float64} = unsafe_wrap(Array, Ccons, ncon)
  _eval_g!(w,cons)
	return 0
end

function eval_grad_f(Cw::Ptr{Cdouble},Cgrad::Ptr{Cdouble}, d::Ptr{Cvoid})::Cint
	w::Vector{Float64} = unsafe_wrap(Array, Cw, nvar)
	grad::Vector{Float64} = unsafe_wrap(Array, Cgrad, nnzo)
  _eval_grad_f!(w,grad)
	@debug "grad-callback" grad
	# Cgrad::Ptr{Cdouble} = unsafe_convert(Ptr{Cdouble}, grad)
	return 0
end

function eval_jac_g(w::Ptr{Cdouble}, Cjac_q::Ptr{Cdouble}, d::Ptr{Cvoid})::Cint
	w::Vector{Float64} = unsafe_wrap(Array, w, nvar)
	jac_g::Vector{Float64} = unsafe_wrap(Array, Cjac_q, nnzj)
  _eval_jac_g!(w,jac_g)
	return 0
end

function eval_h(obj_scale::Cdouble, Cw::Ptr{Cdouble}, Cl::Ptr{Cdouble}, Chess::Ptr{Cdouble}, d::Ptr{Cvoid})::Cint
  w::Vector{Float64} = unsafe_wrap(Array, Cw, nvar)
  l::Vector{Float64} = unsafe_wrap(Array, Cl, ncon)
  hess::Vector{Float64} = unsafe_wrap(Array, Chess, nnzh)
	_eval_h!(w,l,hess)
	return 0
end

nnzo::Int64 = 2
nnzj::Int64 = 2
nnzh::Int64 = 3

nzj_i = Int64[1,1]
nzj_j = Int64[1,2]
nzh_i = Int64[1,1,2]
nzh_j = Int64[1,2,2]
Cnzj_i::Ptr{Int64} = pointer(nzj_i)
Cnzj_j::Ptr{Int64} = pointer(nzj_j)
Cnzh_i::Ptr{Int64} = pointer(nzh_i)
Cnzh_j::Ptr{Int64} = pointer(nzh_j)

x0::Vector{Float64} = [1.0, 1.0]
y0::Vector{Float64} = [1.0]
lbx::Vector{Float64} = [-Inf, -Inf]
ubx::Vector{Float64} = [ Inf,  Inf]
lbg::Vector{Float64} = [0.0]
ubg::Vector{Float64} = [0.0]

max_iters::Int64 = 1000
print_level::Int64 = 3
minimize::Bool = true
user_data::Ptr{Cvoid} = 0

# pre compile for different solvers

lin_solver_names = Dict(
	0=>"mumps",
	1=>"ufpack",
	2=>"lapackCPUsolver",
	3=>"CSSDU",
	4=>"LapackGPUSolver",
	5=>"CuCholeskySolver",
)

sol::Vector{Float64}         = []
obj::Vector{Float64}         = []
con::Vector{Float64}         = []
mul::Vector{Float64}         = []
mul_L::Vector{Float64}       = []
mul_U::Vector{Float64}       = []
iter::Vector{Int64}          = []
primal_feas::Vector{Float64} = []
dual_feas::Vector{Float64}   = []


cases::Vector{Tuple{Int,Int,Int}} = [(0,0,10)]
# cases::Vector{Tuple{Int,Int,Int}} = [(0,3,10),(2,2,1000),(1,1,1000),(0,0,1000)]
for (lin_solver_id,print_level, max_iters) in cases
	nlp_interface = MadnlpCInterface(
		@cfunction(eval_f,Cint,(Ptr{Cdouble},Ptr{Cdouble},Ptr{Cvoid})),
		@cfunction(eval_g,Cint,(Ptr{Cdouble},Ptr{Cdouble},Ptr{Cvoid})),
		@cfunction(eval_grad_f,Cint,(Ptr{Cdouble},Ptr{Cdouble},Ptr{Cvoid})),
		@cfunction(eval_jac_g,Cint,(Ptr{Cdouble},Ptr{Cdouble},Ptr{Cvoid})),
		@cfunction(eval_h,Cint,(Cdouble, Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble},Ptr{Cvoid})),
		nvar,
		ncon,
		Cnzj_i,
		Cnzj_j,
		Cnzh_i,
		Cnzh_j,
		nnzj,
		nnzh,
		nnzo,
		user_data
	)

	s = madnlp_c_create(unsafe_convert(Ptr{MadnlpCInterface}, pointer_from_objref(nlp_interface)))

	out = MadnlpCNumericOut()

	in_c_ptr = madnlp_c_input(s)
	in_c = unsafe_load(in_c_ptr)

	copyto!(unsafe_wrap(Array, in_c.x0, (nvar,)), x0)
	copyto!(unsafe_wrap(Array, in_c.l0, (ncon,)),  y0)
	copyto!(unsafe_wrap(Array, in_c.lbx, (nvar,)), lbx)
	copyto!(unsafe_wrap(Array, in_c.ubx, (nvar,)), ubx)
	copyto!(unsafe_wrap(Array, in_c.lbg, (ncon,)), lbg)
	copyto!(unsafe_wrap(Array, in_c.ubg, (ncon,)), ubg)
	
	madnlp_c_set_option_int(s, unsafe_convert(Ptr{Int8},"lin_solver_id"), lin_solver_id)
	madnlp_c_set_option_int(s, unsafe_convert(Ptr{Int8},"max_iters"), max_iters)
	madnlp_c_set_option_int(s, unsafe_convert(Ptr{Int8},"print_level"), print_level)
	madnlp_c_set_option_bool(s, unsafe_convert(Ptr{Int8},"minimize"), Int64(minimize))

	Cret = madnlp_c_solve(s)

	out_c_ptr = madnlp_c_output(s)
	out_c = unsafe_load(out_c_ptr)

	stats_c_ptr = madnlp_c_get_stats(s)
	stats_c = unsafe_load(stats_c_ptr)

	global sol = unsafe_wrap(Array, out_c.sol, nvar)
	global obj = unsafe_wrap(Array, out_c.obj, ncon)
	global con = unsafe_wrap(Array, out_c.con, ncon)
	global mul = unsafe_wrap(Array, out_c.mul, ncon)
	global mul_L = unsafe_wrap(Array, out_c.mul_L, nvar)
	global mul_U = unsafe_wrap(Array, out_c.mul_U, nvar)

	println("ret_code: ", Cret)
	println("linear_solver: ", lin_solver_names[lin_solver_id])

  println("obj: ", obj[1])
	println("sol: ", sol)
	println("con: ", con)
	println("mul: ", mul)
	println("mul_L: ", mul_L)
	println("mul_U: ", mul_U)
  println("primal_feas: ", primal_feas[1])
  println("dual_feas: ", dual_feas[1])

  println("iter: ", iter[1])

end
