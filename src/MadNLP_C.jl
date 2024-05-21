module MadNLP_C

using MadNLP, JuMP

Base.@ccallable function julia_rosenbrock(x0::Cdouble, y0::Cdouble, z0::Cdouble, iters::Csize_t)::Cint
	model = Model(()->MadNLP.Optimizer(print_level=MadNLP.INFO, max_iter=convert(Int64,iters)))
	@variable(model, x, start = x0)
	@variable(model, y, start = y0)
	@variable(model, z, start = z0)
	@NLobjective(model, Min, x^2 + 100*z^2)
	@NLconstraint(model, z + (1-x)^2 - y == 0)
	optimize!(model)

	println(value(x))
	println(value(y))
	println(value(z))

	return 0
end

end # module Rosenbrock
