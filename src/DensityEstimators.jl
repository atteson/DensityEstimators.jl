module DensityEstimators

using Distributions
using StatsBase
using DSP

export BruteForceKernelDensityEstimator, FFTKernelDensityEstimator

struct BruteForceKernelDensityEstimator{T <: Distribution{Univariate,Continuous}} <: Distribution{Univariate,Continuous}
    dist::T
    h::Float64
    data::Vector{Float64}
end

BruteForceKernelDensityEstimator( dist::T, h::Float64 ) where {T} = BruteForceKernelDensityEstimator( dist, h, Float64[] )

StatsBase.fit( estimator::BruteForceKernelDensityEstimator{T}, data::AbstractVector{Float64} ) where {T} =
    BruteForceKernelDensityEstimator( estimator.dist, estimator.h, data )

Distributions.pdf( kde::BruteForceKernelDensityEstimator{T}, x::Float64 ) where {T <: Distribution{Univariate,Continuous}} =
    mean( pdf.( kde.dist, (x .- kde.data)./kde.h )./kde.h )
    
Distributions.cdf( kde::BruteForceKernelDensityEstimator{T}, x::Float64 ) where {T <: Distribution{Univariate,Continuous}} =
    mean( cdf.( kde.dist, (x .- kde.data)./kde.h ) )

struct FFTKernelDensityEstimator{T <: Distribution{Univariate,Continuous}} <: Distribution{Univariate,Continuous}
    dist::T
    h::Float64
    data::Vector{Float64}
    centers::LinRange{Float64,Int64}
    estimates::Vector{Float64}
end

FFTKernelDensityEstimator( dist::T, h::Float64 ) where {T} =
    FFTKernelDensityEstimator( dist, h, Float64[], LinRange( 0.0, 0.0, 0 ), Float64[] );

function StatsBase.fit( estimator::FFTKernelDensityEstimator{T}, data::AbstractVector{Float64};
                        bits = 14,
                        ) where {T}
    dist = estimator.dist
    h = estimator.h

    n = length(data)
    (m,M) = extrema(data)
    N = 2^bits
    M += eps(M) # histograms are left closed
    range = LinRange( m, M, N+1 )
    centers = (range[1:end-1] + range[2:end])/2
    hist = fit( Histogram, data, range )
    @assert( sum(hist.weights) == n )
    w = hist.weights./n;

    # choose width of convolution kernel; assume symmetry
    tolerance = eps(pdf( dist, 0.0 ))
    dx = (range.stop - range.start)/range.len
    factor = 1
    while pdf( dist, factor*dx/h )/h > tolerance
        factor *= 2
    end

    bound = factor*dx
    range2 = LinRange(-bound, bound, factor*2)
    centers2 = (range2[1:end-1] + range2[2:end])/2
    d = pdf.( dist, centers2./h )./h;
    
    estimates = conv( d, w )[factor:factor+N-1];
    @assert(!any(estimates.<0))

    return FFTKernelDensityEstimator( dist, h, data, centers, estimates )
end

function Distributions.pdf( kde::FFTKernelDensityEstimator{T}, x::Float64 ) where {T <: Distribution{Univariate,Continuous}}
    centers = kde.centers
    estimates = kde.estimates
    r = searchsorted( centers, x )
    if( r.start == 0 || r.stop > length(centers) )
        error( "Out of range!" )
    end
    lambda = (x - centers[r.stop])/(centers[r.start] - centers[r.stop])
    return estimates[r.stop] + lambda * (estimates[r.start] - estimates[r.stop])
end

end
