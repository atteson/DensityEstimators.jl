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

x = [-1.0,1.0]

bf = fit( bf, x )
r = -2:0.001:2
p = histogram( x, normalize=:pdf, fillalpha=0.0, size=[1000,800] )
@time bfy = pdf.( bf, r );
plot!( p, r, bfy, label="" )

fft = FFTKernelDensityEstimator( dist, h )
fft = fit( fft, x )

@time ffty = pdf.( fft, r );
plot!( p, r, ffty, label="", size=[1000,800] )

p = plot( centers[N >> 1:end-N >> 1], estimates )
plot!( p, r, bfy )

emp = ecdf( x )
p = plot( r, emp.(r), label="", size=[1000,800] )
@time y = cdf.( kde, r )
plot!( p, r, y, label="" )

hist = fit( Histogram, x, r )



using DSP
data = x
bits = 14


    n = length(data)
    (m,M) = extrema(data)
    N = 2^bits
    M += eps(M) # histograms are left closed
    range = LinRange( m, M, N+1 )
    hist = fit( Histogram, data, range )
    @assert( sum(hist.weights) == n )
    w = hist.weights./n

    range2 = LinRange((3*m-M)/2, (3*M-m)/2, 2*N)
    centers = (range2[1:end-1] + range2[2:end])/2
    d = pdf.( dist, centers./h )./h

    estimates = conv( d, w )[N:end-N+1] # center could be off by 1
    @assert(!any(estimates.<0))

    Nover2 = N >> 1
    plot( collect(centers[Nover2:end-Nover2]), estimates )
