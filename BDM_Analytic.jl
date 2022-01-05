module BDMAnalytic

    export BDM, prob, SSprob

    using Plots, Parameters, LinearAlgebra, Distributions, StatsBase, DoubleFloats

    """
    Define the binary decision model struct:
        * N: the number of agents in the system.
        * pars: the parameters of the system. Note the length of pars specifies the model type, can be [ϵ,μ], [ϵ1,ϵ2,μ] or [ϵ1,ϵ2,μ1,μ2].
    """
    @with_kw struct BDM
        N::Int64 = 100;
        pars::Vector{Float64}; # can be [ϵ,μ], [ϵ1,ϵ2,μ] or [ϵ1,ϵ2,μ1,μ2]
        model_type::String = "ant" # be also be "voter" or "brock"
        A::Matrix{Float64} = make_TRM(pars, N, model_type);
        λ::Array{Complex{Double64}} = GetEigVals(A,pars,N,model_type);
        q_arr::Vector{Vector{Complex{Double64}}} = [GetOrthoQ(pars,N,λ[j],model_type) for j in 1:N+1];
        p_arr::Vector{Vector{Complex{Double64}}} = [GetOrthoP(pars,N,λ[j],model_type) for j in 1:N+1];
        den_prod::Vector{Complex{Double64}} = [prod([λ[i]-λ[j] for j in filter!(e->e≠i,[j for j in 1:N+1])]) for i in 1:N+1];
        As::Vector{Double64} = [a(pars,N,j,model_type) for j in 1:N];
        Bs::Vector{Double64} = [b(pars,N,j,model_type) for j in 0:N-1];
    end

    """
    Define rate function for ants going to the rh food source
    """
    function a(pars::Vector{Float64}, N::Int64, n::Int64, model_type::String)
        if model_type == "ant"
            if length(pars)==2
                ϵ = pars[1];
                μ = pars[2];
                return convert(Double64,(N-(n-1))*ϵ+μ*(n-1)*(N-(n-1))/(N-1))::Double64
            elseif length(pars)==3
                ϵ1 = pars[1];
                μ = pars[3];
                return convert(Double64,(N-(n-1))*ϵ1+μ*(n-1)*(N-(n-1))/(N-1))::Double64
            elseif length(pars)==4
                ϵ1 = pars[1];
                μ1 = pars[3];
                return convert(Double64,(N-(n-1))*ϵ1+μ1*(n-1)*(N-(n-1))/(N-1))::Double64
            else
                error("length of pars must be 2, 3 or 4.")
            end
        elseif model_type == "voter"
            if length(pars)==1
                p₀ = pars[1];
                if p₀ > 1 || p₀ < 0
                    error("p₀ must be between 0 and 1.")
                end
                ϵ = p₀/(2*N);
                μ = (1-p₀)*(N-1)/(2*N^2);
                return convert(Double64,(N-(n-1))*ϵ+μ*(n-1)*(N-(n-1))/(N-1))::Double64
            elseif length(pars)==2
                ϵ = pars[1];
                μ = pars[2];
                return convert(Double64,(N-(n-1))*ϵ+μ*(n-1)*(N-(n-1))*(1+(N-(n-1))/(N-1))/(N-1))::Double64
            else
                error("length of pars must be 1 or 2 for the voter models")
            end
        elseif model_type == "brock"
            if length(pars)==5
                α = pars[1];
                β = pars[2];
                γ = pars[3];
                F = pars[4];
                J = pars[5];
                mn = (2*(n-1)-N)/N;
                return convert(Double64,(N-(n-1))*γ/(1+exp(-β*(F+J*(α+1)*mn))))
            else
                error("length of pars must be 5 for the brock and durlauf model")
            end
        else
            error("chosen model type is not recognised")
        end
    end

    """
    Define rate function for ants going to the lh food source
    """
    function b(pars::Vector{Float64}, N::Int64, n::Int64, model_type::String)
        if model_type == "ant"
            if length(pars)==2
                ϵ = pars[1];
                μ = pars[2];
                return convert(Double64,(n+1)*ϵ+μ*(n+1)*(N-(n+1))/(N-1))::Double64
            elseif length(pars)==3
                ϵ2 = pars[2];
                μ = pars[3];
                return convert(Double64,(n+1)*ϵ2+μ*(n+1)*(N-(n+1))/(N-1))::Double64
            elseif length(pars)==4
                ϵ2 = pars[2];
                μ2 = pars[4];
                return convert(Double64,(n+1)*ϵ2+μ2*(n+1)*(N-(n+1))/(N-1))::Double64
            else
                error("length of pars must be 2, 3 or 4.")
            end
        elseif model_type == "voter"
            if length(pars)==1
                p₀ = pars[1];
                ϵ = p₀/(2*N);
                μ = (1-p₀)*(N-1)/N^2;
                return convert(Double64,(n+1)*ϵ+μ*(n+1)*(N-(n+1))/(N-1))::Double64
            elseif length(pars)==2
                ϵ = pars[1];
                μ = pars[2];
                return convert(Double64,(n+1)*ϵ+μ*(n+1)*(N-(n+1))*(1+(N-(n+1))/(N-1))/(N-1))::Double64
            else
                error("length of pars must be 1 or 2 for the voter models")
            end
        elseif model_type == "brock"
            if length(pars)==5
                α = pars[1];
                β = pars[2];
                γ = pars[3];
                F = pars[4];
                J = pars[5];
                mn = (2*(n+1)-N)/N;
                return convert(Double64,(n+1)*γ/(1+exp(-β*(F+J*(α+1)*mn))))
            else
                error("length of pars must be 5 for the brock and durlauf model")
            end
        else
            error("chosen model type is not recognised")
        end
    end

    """
    Make the transition rate matrix
    """
    function make_TRM(pars::Vector{Float64}, N::Int64, model_type::String)
        A = zeros(N+1, N+1);
        for i in 1:size(A)[1]
            for j in 1:size(A)[2]
                if i == 1 && j == 1
                    A[1,1] = -a(pars,N,1, model_type)
                elseif i == j && i>1
                    A[i,i] = -(a(pars,N,i, model_type)+b(pars,N,i-2, model_type))
                elseif i == j+1
                    A[i,j] = a(pars,N,j, model_type)
                elseif i == j-1
                    A[i,j] = b(pars,N,i-1, model_type)
                else
                    continue
                end
            end
        end
        return A::Matrix{Float64}
    end

    """
    Get eigenvalues
    """
    function GetEigVals(A::Matrix{Float64},pars::Vector{Float64},N::Int64,model_type::String)
        if model_type == "ant"
            if length(pars)==2
                ϵ = pars[1];
                μ = pars[2];
                λ = [-(m-1)*(2*ϵ+(m-2)μ/(N-1)) for m in 1:N+1];
                return convert(Array{Complex{Double64}},λ)
            elseif length(pars)==3
                ϵ1 = pars[1];
                ϵ2 = pars[2]
                μ = pars[3];
                λ = [-(m-1)*(ϵ1+ϵ2+(m-2)μ/(N-1)) for m in 1:N+1];
                return convert(Array{Complex{Double64}},λ)
            elseif length(pars)==4
                λ = convert(Array{Complex{Double64}}, reverse(eigvals(A)));
                if λ[1] == λ[2] # if get repeated zero eigenvalues from solver manually separate.
                    λ[1] = 0.0 + (1E-30)im; λ[2] = 0.0 - (1E-30)im;
                end
                return λ::Array{Complex{Double64}}
            else
                error("length of pars must be 2, 3 or 4.")
            end
        elseif model_type == "voter"
            if length(pars)==1
                p₀ = pars[1];
                ϵ = p₀/(2*N);
                μ = (1-p₀)*(N-1)/N^2;
                λ = [-(m-1)*(2*ϵ+(m-2)μ/(N-1)) for m in 1:N+1];
                return convert(Array{Complex{Double64}},λ)
            elseif length(pars)==2
                λ = convert(Array{Complex{Double64}}, reverse(eigvals(A)));
                if λ[1] == λ[2] # if get repeated zero eigenvalues from solver manually separate.
                    λ[1] = 0.0 + (1E-30)im; λ[2] = 0.0 - (1E-30)im;
                end
                return λ::Array{Complex{Double64}}
            else
                error("length of pars must be 1 or 2 for voter models")
            end
        elseif model_type == "brock"
            if length(pars)==5
                α = pars[1];
                β = pars[2];
                γ = pars[3];
                F = pars[4];
                J = pars[5];
                λ = [-(m-1)*(N-β*J*(α+1)*(2+N-m))*γ/N for m in 1:N+1];
                return convert(Array{Complex{Double64}},λ)
            else
                error("length of pars must be 5 for brock model")
            end
        end
    end

    """
    Get the p orthogonal polynomials
    """
    function GetOrthoP(pars::Vector{Float64}, N::Int64, λᵢ::Complex{Double64},model_type::String)
        p = Array{Complex{Double64},1}(undef, N+1);
        p[1] = 1.0; p[2] = λᵢ+a(pars,N,1,model_type);
        for i in 3:N+1
            p[i] = (λᵢ+a(pars,N,i-1,model_type)+b(pars,N,i-3,model_type))*p[i-1] - b(pars,N,i-3,model_type)*a(pars,N,i-2,model_type)*p[i-2]
        end
        return p::Vector{Complex{Double64}}
    end

    """
    Get the q orthogonal polynomials
    """
    function GetOrthoQ(pars::Vector{Float64}, N::Int64, λᵢ::Complex{Double64},model_type::String)
        q = Array{Complex{Double64},1}(undef, N+3);
        q[N+3] = 1.0; q[N+2] = λᵢ + b(pars,N,N-1,model_type);
        for i in reverse([j for j in 3:N+1])
            q[i] = (λᵢ + a(pars,N,i-1,model_type)+b(pars,N,i-3,model_type))*q[i+1] - b(pars,N,i-2,model_type)*a(pars,N,i-1,model_type)*q[i+2]
        end
        return q::Vector{Complex{Double64}}
    end

    """
    Define the sum of the elements from the solution
    """
    function sum_elems(λᵢ::Complex{Double64}, t::Float64, m::Int64, m₀::Int64, pars::Array{Float64,1}, N::Int64, p_arrᵢ::Vector{Complex{Double64}}, q_arrᵢ::Vector{Complex{Double64}}, den_prodᵢ::Complex{Double64})
        return exp(λᵢ*t)*p_arrᵢ[m+1]*q_arrᵢ[m₀+3]/den_prodᵢ::Complex{Double64}
    end

    """
    Define P(x,t|x₀)
    """
    function pm(t::Float64, BD::BDM, m::Int64, m₀::Int64)
        @unpack λ, pars, N, As, Bs, q_arr, p_arr, den_prod = BD
        if m<m₀
            return convert(Double64,N)*prod(Bs[m+1:m₀])*sum([sum_elems(λ[i], t, m, m₀, pars, N, p_arr[i], q_arr[i], den_prod[i]) for i in 1:N+1])::Complex{Double64}
        elseif m==m₀
            return convert(Double64,N)*sum([sum_elems(λ[i], t, m, m, pars, N, p_arr[i], q_arr[i], den_prod[i]) for i in 1:N+1])::Complex{Double64}
        else
            return convert(Double64,N)*prod(As[m₀+1:m])*sum([sum_elems(λ[i], t, m₀, m, pars, N, p_arr[i], q_arr[i], den_prod[i]) for i in 1:N+1])::Complex{Double64}
        end
    end

    """
    Define the probability distribution return function from a initial distribution.
    """
    function prob(BD::BDM, t::Float64, q_init_D::Distribution{Univariate, Discrete})
        @unpack N = BD
        q_init = pdf(q_init_D)
        pmt = Array{Complex{Double64}}(undef,N+1)
        for i in 1:N+1 # loop over the m0's
            pmt[i] = sum([q_init[n+1]*pm(t, BD, i-1, n) for n in 0:N])
        end
        return (LinRange(-1.0,1.0,N+1),real(pmt))::Tuple{LinRange{Float64}, Vector{Double64}}
    end

    """
    Define the probability distribution return function from a precise value of m₀.
    """
    function prob(BD::BDM, t::Float64, m₀::Int64)
        @unpack N = BD
        return (LinRange(-1.0,1.0,N+1),real([pm(t, BD, m, m₀) for m in 0:N]))::Tuple{LinRange{Float64}, Vector{Double64}}
    end

    """
    Define the probability distribution return function for N/2 value of m₀.
    """
    function prob(BD::BDM, t::Float64)
        @unpack N = BD
        m₀ = floor(Int64,N/2) # note that we have the n=0 state too.
        return (LinRange(-1.0,1.0,N+1),real([pm(t, BD, m, m₀) for m in 0:N]))::Tuple{LinRange{Float64}, Vector{Double64}}
    end

    """
    Return the steady state distribution
    """
    function SSprob(BD::BDM)
        @unpack As, Bs, N = BD
        ps = Vector{Double64}(undef, N+1)
        for m in 2:N # use the product rule
            ps[m] = prod(As[1:m-1])*prod(Bs[m:N])
        end
        ps[1] = prod(Bs[1:N]) # do product for the B's
        ps[N+1] = prod(As[1:N]) # do product for the A's
        return (LinRange(-1.0,1.0,N+1),N.*ps/sum(ps))::Tuple{LinRange{Float64}, Vector{Double64}}
    end

end # module end
