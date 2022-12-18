using DensityEstimators
using Random
using Distributions
using Interpolations

Random.seed!(1)

N = 1_000

dist = Normal()

x = rand( dist, N );

@time kde = FFTKernelDensityEstimator( Normal(), 0.1 );
@time kde = fit( kde, x );

@time k = collect(knots(kde.interpolator));
@time p = collect(kde.interpolator.itp);

kste = maximum(abs.( cdf.( dist, k ) .- cumsum(p)./sum(p) ))
kstx = KS( dist, x )

