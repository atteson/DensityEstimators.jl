
using Distributions
using Random

export CantorDistribution

struct CantorDistribution <: Distribution{Univariate,Continuous}
end

function Distributions.cdf( ::CantorDistribution, x::Float64 )
    x >= 1.0 && return 1.0
    x <= 0.0 && return 0.0
    sum = 0.0
    increment = 1.0
    while increment > eps(sum)
        increment /= 2
        x *= 3
        if x >= 1
            sum += increment
            if x < 2
                return sum
            end
        end
        x %= 1
    end
    return sum
end

function upperinverseCantor( q::Float64 )
    sum = 0.0
    increment = 2.0
    while increment > eps(sum)
        increment /= 3
        q *= 2
        if q >= 1 
            sum += increment
        end
        q %= 1
    end
    return sum
end

Distributions.quantile( ::CantorDistribution, q::Float64 ) =
    1 - upperinverseCantor( 1 - q )

Base.rand( rng::AbstractRNG, c::CantorDistribution ) =
    quantile( c, rand( rng ) )
