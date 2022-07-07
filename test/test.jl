using DensityEstimators
using Random
using Distributions
using Plots
using StatsBase

dist = Normal()
h = 0.1
bf = BruteForceKernelDensityEstimator( dist, h )

N = 100
Random.seed!(1)
x = rand( dist, N );

@time bf = fit( bf, x );
r = -2:0.001:2
p = histogram( x, normalize=:pdf, fillalpha=0.0, size=[1000,800] )
@time bfy = pdf.( bf, r );
plot!( p, r, bfy, label="" )

fft = FFTKernelDensityEstimator( dist, h )
@time fft = fit( fft, x, bits=20 );

@time ffty = pdf.( fft, r );
plot!( p, r, ffty, label="", size=[1000,800] )

maximum(abs.(bfy - ffty))
mean(abs.(bfy - ffty))
sqrt(mean((bfy - ffty).^2))

emp = ecdf( x )
p = plot( r, emp.(r), label="", size=[1000,800] )
@time y = cdf.( bf, r )
plot!( p, r, y, label="" )
