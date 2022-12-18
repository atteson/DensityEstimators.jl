using DensityEstimators
using Random
using Distributions
using Plots
using StatsBase
using Dates

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
plot( collect(hes), kses, ylims=[0,1], label="", xaxis=:log, leftmargin=30*Plots.px )

findmin(kses)


hes = 2.0 .^ (-20:3)
kses = Vector{Float64}[]
N = 1_000
@time for h in hes
    println( "Running h = $h at $(now())" )
    bf = BruteForceKernelDensityEstimator(dist,h)
    bf = fit( bf, x )
    ks = Float64[]
    for i = 1:N
        y = rand( bf, n )
        push!( ks, KS( bf, y ) )
    end
    push!( kses, sort(ks) )
end
p = plot( size=[1000,800] )
[plot!( p, ks, collect((1:N)/N), label="" ) for ks in kses]
display(p)

r = 0.006:0.001:0.2
@time exact = noeks.( dist, n, r );
plot!( p, collect(r), exact, label="", linecolor=:black, linewidth=2 )
display(p)

@time exact2 = noeks.( Uniform(), n, r );
@assert(minimum(abs.(exact - exact2)) < 1e-8)

savefig( p, joinpath( homedir(), "invariance.png" ) )

function pvalue( dist, h, x, N; refit = false, actual = false )
    bf = BruteForceKernelDensityEstimator( dist, h )
    bf = fit( bf, x )
    ks = KS( bf, x )
    count = 0
    n = length(x)
    for i = 1:N
        y = rand( actual ? dist : bf, n )
        if refit
            bf1 = fit( bf, y )
        elseif actual
            bf1 = dist
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
@time qs = pvalue.( dist, r, [x], 1_000, refit=true );
@time as = pvalue.( dist, r, [x], 1_000, actual=true );

p = plot( r, ps, label="simulated p-value", size=[1000,800], leftmargin=20*Plots.px )
plot!( p, r, qs, label="refit p-value" )
plot!( p, r, as, label="actual p-value" )

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


p = plot( 0:0.001:1, 0:0.001:1, label="exact", size=[1000,800], framestyle=:box, ylims=[0,1], xlims=[0,1], legend_position=:bottomright )
u = Uniform()
for n in [10, 20, 50, 100, 200]
    r0 = 1/(2*n) + eps(Float64)
    prob = noeks( u, n, r0 )
    rinf = r0
    while prob[end] < 0.999
        rinf *= 2
        prob = noeks( u, n, rinf )
    end
    r = r0:(rinf-r0)/1000:rinf
    @time exact = noeks.(u, n, r );
    @time kolmogorov = cdf.( Kolmogorov(), sqrt(n).*r );
    plot!( p, exact, kolmogorov, label="$n" )
end
display(p)

