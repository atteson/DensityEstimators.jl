export noe, noeks

using SpecialFunctions

Base.binomial( n::T, m::T ) where {T <: AbstractFloat} =
    exp(loggamma( n + 1 ) - (loggamma( m + 1 ) + loggamma( n - m + 1 )))

function noe( dist, a::AbstractVector{T}, b::AbstractVector{T} ) where {T <: AbstractFloat}
    @assert( issorted( a ) )
    @assert( issorted( b ) )
    n = length(a)
    @assert( n == length(b) )
    @assert( all( a .< b ) )

    c = fill( T(NaN), 2*(n+1) )
    c[1] = -Inf
    c[end] = Inf

    g = fill( 0, 2*n+1 )
    h = fill( 1, 2*n+1 )

    i = 1
    j = 1
    k = 2
    while i <= n
        if a[i] <= b[j]
            c[k] = a[i]
            i += 1
        else
            c[k] = b[j]
            j += 1
        end
        g[k] = i-1
        h[k] = j
        k += 1
    end
    while j <= n
        c[k] = b[j]
        j += 1
        g[k] = i-1
        h[k] = j
        k += 1
    end
    
    p = diff( cdf.( dist, c ) )

    Q = zeros( T, n+1, 2*n+1 )
    Q[1,1] = 1
    for m = 1:2*n
        for i = (h[m+1]-1):g[m]
            s = T(0.0)
            for k = (h[m]-1):i
                s += binomial( T(i), T(k) ) * Q[ k+1, m ] * p[m]^(i-k)
            end
            Q[i+1, m+1] = s
        end
    end
    return Q[n+1, 2*n+1]
end

noe( dist, a::AbstractVector{T}, b::AbstractVector{T} ) where {T <: Integer} = noe( dist, Float64.(a), Float64.(b) )

function noeks( dist, n, x )
    a = quantile.( dist, max.((1:n)/n .- x,0) )
    b = quantile.( dist, min.((0:n-1)/n .+ x,1) )
    return noe( dist, a, b )
end


