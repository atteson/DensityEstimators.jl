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

s=[1000,800]
s=[600,500]
@time bf = fit( bf, x );
r = -2:0.001:2
p = histogram( x, normalize=:pdf, fillalpha=0.0, size=s )
@time bfy = pdf.( bf, r );
plot!( p, r, bfy, label="" )

fft = FFTKernelDensityEstimator( dist, h )
@time fft = fit( fft, x );
@time ffty = pdf.( fft, r );
plot!( p, r, ffty, label="" )

sqrt(mean((bfy .- ffty).^2))

bits = 10:20
accuracies = fill( NaN, 3, length(bits) )
for i=1:length(bits)
    @time fft = fit( fft, x, bits=bits[i] );
    @time ffty = pdf.( fft, r );

    accuracies[1,i] = mean(abs.(bfy - ffty))
    accuracies[2,i] = sqrt(mean((bfy - ffty).^2))
    accuracies[3,i] = maximum(abs.(bfy - ffty))
end

emp = ecdf( x )
p = plot( r, emp.(r), label="", size=s )
@time y = cdf.( bf, r )
plot!( p, r, y, label="" )

c = CantorDistribution();
N = 1_000_000
Random.seed!(1)
@time x = rand( c, N );
@assert( !any((x .> 1/3) .& (x .< 2/3)) );
@assert( !any((x .> 1/9) .& (x .< 2/9)) );
@assert( !any((x .> 7/9) .& (x .< 8/9)) );

N = 100
Random.seed!(1)
@time x = rand( c, N );
@time bf = fit( bf, x );
r = 0.001:0.001:0.999
@time bfy = pdf.( bf, r );

@time fft = fit( fft, x );
@time ffty = pdf.( fft, r );

sqrt(mean((bfy .- ffty).^2))

N = 100
Random.seed!(1)
x = rand( dist, N );
r = -2:0.001:2

hs = 10 .^ -1:6
accuracies = fill( NaN, 3, length(hs) )
for i = 1:length(hs)
    bf = BruteForceKernelDensityEstimator( dist, hs[i] )
    @time bf = fit( bf, x );
    @time bfy = pdf.( bf, r );

    fft = FFTKernelDensityEstimator( dist, hs[i] )
    @time fft = fit( fft, x )
    @time ffty = pdf.( fft, r );

    accuracies[1,i] = mean(abs.(bfy - ffty))
    accuracies[2,i] = sqrt(mean((bfy - ffty).^2))
    accuracies[3,i] = maximum(abs.(bfy - ffty))
end

dist = Normal()
h = 1e-6
Random.seed!(1)

x = rand( dist, 100 );
bf = BruteForceKernelDensityEstimator(dist,h)
bf = fit( bf, x )

DensityEstimators.KS( bf, x )
