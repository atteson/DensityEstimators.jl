using DensityEstimators
using Random
using Distributions
using Plots
using StatsBase

dist = Normal()
h = 1e-4
Random.seed!(1)

x = rand( dist, 100 );
hes = 2 .^ (-20:0.1:3)
kses = Float64[]
@time for h in hes
    bf = BruteForceKernelDensityEstimator(dist,h)
    bf = fit( bf, x )
    push!( kses, KS( bf, x ) )
end
plot( collect(hes), kses, label="", yaxis=:log, xaxis=:log )

findmin(kses)

h = hes[1]
bf = BruteForceKernelDensityEstimator(dist,h)
bf = fit( bf, x )

samplekses = Float64[]
for i = 1:1000
    y = rand( bf, 100 )
    push!( samplekses, KS( bf, y ) )
end
sum( samplekses .< kses[1] )
sum( samplekses .> kses[1] )
