export noi

function noi( dist, a, b )
    @assert( issorted( a ) )
    @assert( issorted( b ) )
    n = length(a)
    @assert( n == length(b) )
    @assert( all( a .< b ) )

    c = fill( NaN, 2*(n+1) )
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

    Q = zeros( n+1, 2*n+1 )
    Q[1,1] = 1
    for m = 1:2*n
        for i = (h[m+1]-1):g[m]
            s = 0.0
            for k = (h[m]-1):i
                s += binomial( i, k ) * Q[ k+1, m ] * p[m]^(i-k)
            end
            Q[i+1, m+1] = s
        end
    end
    return Q[n+1, 2*n+1]
end



