using DensityEstimators
using Distributions
using Random

dist = Normal()

a = [0,2]
b = [1,3]
@assert( abs(noe( dist, a, b ) - 2*prod(cdf.( dist, b ) - cdf.( dist, a ))) < 1e-8 )

a = [0,1]
b = [2,3]
q = diff(cdf.( dist, sort([a;b]) ))
@assert( abs(noe( dist, a, b ) - (2 * (q[1]*q[2] + q[1]*q[3] + q[2]*q[3]) + q[2]^2)) < 1e-8 )

a = [0,1]
b = [2,2]
q = diff(cdf.( dist, sort([a;b]) ))
@assert( abs(noe( dist, a, b ) - (2 * (q[1]*q[2] + q[1]*q[3] + q[2]*q[3]) + q[2]^2)) < 1e-8 )

n = 5
Random.seed!(1)
u = sort(randn(n))
v = sort(3*randn(n))
a = min.( u, v )
b = max.( u, v )

@assert( (v -> any(v) && !all(v))(a[2:end] .> b[1:end-1]) )

@assert( abs(mean([all(a .< sort(rand( dist, length(a) )) .<= b) for i in 1:10_000_000]) / noe( dist, a, b ) - 1) < 0.005 )

@time noeks( dist, 100, 0.1 )

n = 2
x = 0.4
noeks( dist, n, x )

u = min.( (0:n-1)./n .+ x, 1 )
l = max.( (1:n)./n .- x, 0 )

l[1] * l[2] * u[1]*u[2] - u[1]^2 - l[1]*u[2]

noe( dist, quantile.( dist, l ), quantile.( dist, u ) )


n = 1000
x = 0.02
@time exact = noeks( Uniform(), n, x )

@time kolmogorov = cdf( Kolmogorov(), x * sqrt(n) )

abs(kolmogorov/exact - 1)
