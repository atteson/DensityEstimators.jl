
module DensityEstimators

using Random
using Distributions
using StatsBase
using DSP
using Interpolations

export BruteForceKernelDensityEstimator, FFTKernelDensityEstimator

abstract type AbstractKernelDensityEstimator{T <: Distribution{Univariate,Continuous}} <: Distribution{Univariate,Continuous}
end

struct BruteForceKernelDensityEstimator{T} <: AbstractKernelDensityEstimator{T}
    dist::T
    h::Float64
    data::Vector{Float64}
end

BruteForceKernelDensityEstimator( dist::T, h::Float64 ) where {T} = BruteForceKernelDensityEstimator( dist, h, Float64[] )

StatsBase.fit( estimator::BruteForceKernelDensityEstimator{T}, data::AbstractVector{Float64} ) where {T} =
    BruteForceKernelDensityEstimator( estimator.dist, estimator.h, data )

Distributions.pdf( kde::BruteForceKernelDensityEstimator{T}, x::Float64 ) where {T} = 
    mean( pdf.( kde.dist, (x .- kde.data)./kde.h )./kde.h )
    
Distributions.cdf( kde::BruteForceKernelDensityEstimator{T}, x::Float64 ) where {T} =
    mean( cdf.( kde.dist, (x .- kde.data)./kde.h ) )

struct FFTKernelDensityEstimator{T} <: AbstractKernelDensityEstimator{T}
    dist::T
    h::Float64
    data::Vector{Float64}
    interpolator::AbstractInterpolation{Float64,1}
end

FFTKernelDensityEstimator( dist::T, h::Float64 ) where {T} =
    FFTKernelDensityEstimator(
        dist, h, Float64[], LinearInterpolation( Float64[], Float64[] )
    );

function StatsBase.fit( estimator::FFTKernelDensityEstimator{T},
                        data::AbstractVector{Float64};
                        bits = 14,
                        ) where {T}
    dist = estimator.dist
    h = estimator.h

    n = length(data)
    (m,M) = extrema(data)
    N = 2^bits
    M += eps(M) # histograms are left closed
    range = LinRange( m, M, N )
    centers = (range[1:end-1] + range[2:end])/2
    hist = fit( Histogram, data, range )

    interpolator = LinearInterpolation( range, 1:length(range) )
    parts = divrem.( interpolator.( data ), 1 )
    int = Int.(getindex.(parts,1))
    frac = getindex.(parts,2)
    nfrac = 1 .- frac
    test = getindex.( [range], int ) .* nfrac
    test .+= get.( [range], int .+ 1, range[end] ) .* frac
    @assert( maximum(abs.(test .- data)) .< 1e-8 )
    
    weights = zeros( length(range) )
    for i = 1:length(int)
        weights[int[i]] += nfrac[i]
        if int[i] < length(weights)
            weights[int[i]+1] += frac[i]
        end
    end
    
    @assert( abs(sum(weights) - n) < 1e-8 )
    weights ./= n;

    # choose width of convolution kernel; assume symmetry
    dx = (range.stop - range.start)/range.len
    factor = N >> 1

    bound = factor*dx
    range2 = LinRange(-bound, bound, factor*2)
    centers2 = (range2[1:end-1] + range2[2:end])/2
    d = pdf.( dist, centers2./h )./h;
    
    estimates = conv( d, weights )[factor:factor+N-1];
    @assert(!any(estimates.<0))

    return FFTKernelDensityEstimator(
        dist, h, data, LinearInterpolation( range, estimates )
    )
end

Distributions.pdf( kde::FFTKernelDensityEstimator{T}, x::Float64 ) where {T} = kde.interpolator( x )

function Random.rand!( rng::AbstractRNG, kde::AbstractKernelDensityEstimator{T}, v::AbstractArray{Float64} ) where {T}
    for i = 1:length(v)
        v[i] = rand( rng, kde.data ) + kde.h * rand( rng, kde.dist )
    end
end

end
