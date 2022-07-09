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



