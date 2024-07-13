using MadNLP_C
using Base
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
	# Cgrad::Ptr{Cdouble} = Base.unsafe_convert(Ptr{Cdouble}, grad)
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

nnzo::Csize_t = 2
nnzj::Csize_t = 2
nnzh::Csize_t = 3

nzj_i = Csize_t[1,1]
nzj_j = Csize_t[1,2]
nzh_i = Csize_t[1,1,2]
nzh_j = Csize_t[1,2,2]
Cnzj_i::Ptr{Csize_t} = pointer(nzj_i)
Cnzj_j::Ptr{Csize_t} = pointer(nzj_j)
Cnzh_i::Ptr{Csize_t} = pointer(nzh_i)
Cnzh_j::Ptr{Csize_t} = pointer(nzh_j)

x0::Vector{Float64} = [1.0, 1.0]
y0::Vector{Float64} = ones(ncon)
lbx = Cdouble[-Inf, -Inf]
ubx = Cdouble[ Inf,  Inf]
lbg = Cdouble[0.0]
ubg = Cdouble[0.0]

Cx0::Ptr{Cdouble} = pointer(x0)
Cy0::Ptr{Cdouble} = pointer(y0)
Clbx::Ptr{Cdouble} = pointer(lbx)
Cubx::Ptr{Cdouble} = pointer(ubx)
Clbg::Ptr{Cdouble} = pointer(lbg)
Cubg::Ptr{Cdouble} = pointer(ubg)

max_iters::Csize_t = 1000
print_level::Csize_t = 3
minimize::Bool = true
user_data::Ptr{Cvoid} = 0

sol = zeros(nvar)
Csol::Ptr{Cdouble} = pointer(sol)
obj = zeros(1)
Cobj::Ptr{Cdouble} = pointer(obj)
con = zeros(ncon)
Ccon::Ptr{Cdouble} = pointer(con)

primal_feas = zeros(1)
Cprimal_feas::Ptr{Cdouble} = pointer(primal_feas)

dual_feas = zeros(1)
Cdual_feas::Ptr{Cdouble} = pointer(dual_feas)

mul = zeros(ncon)
Cmul::Ptr{Cdouble} = pointer(mul)

mul_L = zeros(nvar)
mul_U = zeros(nvar)
Cmul_L::Ptr{Cdouble} = pointer(mul_L)
Cmul_U::Ptr{Cdouble} = pointer(mul_U)

iter::Vector{Int} = [1,]
Citer::Ptr{Csize_t} = pointer(iter)

# pre compile for different solvers

lin_solver_names = Dict(
	0=>"mumps",
	1=>"ufpack",
	2=>"lapackCPUsolver",
	3=>"CSSDU",
	4=>"LapackGPUSolver",
	5=>"CuCholeskySolver",
)
cases::Vector{Pair{UInt64,Csize_t}} = [0=>3]
# cases::Vector{Pair{UInt64,Csize_t}} = [0=>3,1=>3,5=>3,3=>0]
for (lin_solver_id,print_level) in cases

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

	s = madnlp_c_create(Base.unsafe_convert(Ptr{MadnlpCInterface}, pointer_from_objref(nlp_interface)))

	inp = MadnlpCNumericIn(Cx0,Cy0,Clbx,Cubx,Clbg,Cubg)
	# inp = MadnlpCNumericIn{Ptr{typeof(Cx0)}}()
	# inp.x0  = Cx0
	# inp.l0  = Cy0
	# inp.lbx = Clbx
	# inp.ubx = Cubx
	# inp.lbg = Clbg
	# inp.ubg = Cubg

	@info "lbx" inp.lbx
	out = MadnlpCNumericOut()

	s_jl::MadnlpCSolver = unsafe_load(s)
	s_jl.in_c = inp
	s_jl.out_c = out
	s = Base.unsafe_convert(Ptr{MadnlpCSolver}, pointer_from_objref(s_jl))

  test_x0 = unsafe_wrap(Array, inp.x0, (nvar,))
	@info test_x0

	#madnlp_c_set_option_int(s, "lin_solver_id", lin_solver_id)
	#madnlp_c_set_option_int(s, "max_iters", max_iters)
	#madnlp_c_set_option_int(s, "print_level", print_level)
	#madnlp_c_set_option_bool(s, "minimize", minimize)
	#

	Cret = madnlp_c_solve(s)
					# Base.unsafe_convert(Ptr{MadnlpCNumericIn}, pointer_from_objref(inp)),
					# Base.unsafe_convert(Ptr{MadnlpCNumericOut}, pointer_from_objref(out)))

	@info "sol" out.sol
	sol_out::Vector{Float64} = unsafe_wrap(Array, out.sol, nvar)
	obj_out::Vector{Float64} = unsafe_wrap(Array, out.obj, ncon)
	con_out::Vector{Float64} = unsafe_wrap(Array, out.con, ncon)
	mul_out::Vector{Float64} = unsafe_wrap(Array, out.mul, ncon)
	mull_out::Vector{Float64} = unsafe_wrap(Array, out.mul_L, nvar)
	mulu_out::Vector{Float64} = unsafe_wrap(Array, out.mul_U, nvar)

	println("ret_code: ", Cret)

	println("linear_solver: ", lin_solver_names[lin_solver_id])
	println("sol: ", sol_out)
	println("con: ", con_out)
	println("obj: ", obj_out[1])
	println("mul: ", mul_out)
	println("mul_L: ", mull_out)
	println("mul_U: ", mulu_out)

end
