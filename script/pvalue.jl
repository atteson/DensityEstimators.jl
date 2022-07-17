using DensityEstimators
using Random
using Distributions
using Plots
using StatsBase

dist = Normal()
Random.seed!(1)

n = 100

x = rand( dist, n );
hes = 2 .^ (-20:0.1:3)
kses = Float64[]
@time for h in hes
    bf = BruteForceKernelDensityEstimator(dist,h)
    bf = fit( bf, x )
    push!( kses, KS( bf, x ) )
end
plot( collect(hes), kses, label="", yaxis=:log, xaxis=:log )

findmin(kses)

function pvalue( dist, h, x, N; refit = false )
    bf = BruteForceKernelDensityEstimator(dist,h)
    bf = fit( bf, x )
    ks = KS( bf, x )
    count = 0
    n = length(x)
    for i = 1:N
        y = rand( bf, n )
        if refit
            bf1 = fit( bf, y )
        else
            bf1 = bf
        end
        count += KS( bf1, y ) > ks
    end
    return count/N
end

h = hes[1]
@time p = pvalue( dist, h, x, 100_000 );

r = 1.005 .^ (-250:150);

@time ps = pvalue.( dist, r, [x], 1_000 );
[r ps]
p = plot( r, ps, label="p-value" )

@time qs = pvalue.( dist, r, [x], 1_000, refit=true );
plot!( p, r, qs, label="refit p-value" )

savefig( "p-values.png" )

kses = [KS( dist, rand( dist, n ) ) for i in 1:1_000];
mean(KS( dist, x ) .<= kses)

function findh( dist, x, p; significance=0.01, tolerance=0.01, h0 = 1.0 )
    n = length(x)
    q = -quantile( Normal(), significance/2 )
    n = Int(ceil(p*(1-p)*(q/tolerance)^2))

    h = h0
    hip = 0.0
    lop = 1.0
    hih = 0.0
    loh = Inf
    currp = pvalue( dist, h, x, n )
    while lop > p || hip < p || abs(currp - p) < tolerance
        if currp < p
            lop = currp
            loh = h
            h = h/2
        else
            hip = currp
            hih = h
            h = h*2
        end
        currp = pvalue( dist, h, x, n )
    end

    while abs(currp - p) >= tolerance
        h = (loh + hih)/2
        currp = pvalue( dist, h, x, n )
        if currp > p
            hip = currp
            hih = h
        else
            lop = currp
            loh = h
        end
    end
    return h
end

Random.seed!(1)
findh( dist, x, 0.5 )


