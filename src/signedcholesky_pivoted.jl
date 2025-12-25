module SignedCholPiv

##########################
# Signed Cholesky Factorization #
##########################

using LinearAlgebra 

import LinearAlgebra: checksquare, RealHermSymComplexHerm, 
        BlasInt, checknonsingular, eigencopy_oftype
import Base: require_one_based_indexing, copy, show

export signedcholesky, signedcholesky!

# --------------  Pivoted Signed Cholesky ----------------


struct SignedCholeskyPivoted{T,S<:AbstractMatrix,P<:AbstractVector{<:Integer}} <: Factorization{T}
    factors::S
    signs::Vector{Int8}
    uplo::Char
    piv::P
    rank::BlasInt
    tol::Real
    info::BlasInt

    function SignedCholeskyPivoted{T,S,P}(factors, signs, uplo, piv, rank, tol, info) where {T,S<:AbstractMatrix,P<:AbstractVector}
        require_one_based_indexing(factors)
        new{T,S,P}(factors, signs, uplo, piv, rank, tol, info)
    end
end

SignedCholeskyPivoted(A::AbstractMatrix{T}, uplo::AbstractChar, piv::AbstractVector{<:Integer},
                rank::Integer, tol::Real, info::Integer) where T =
    SignedCholeskyPivoted{T,typeof(A),typeof(piv)}(A, uplo, piv, rank, tol, info)
# backwards-compatible constructors (remove with Julia 2.0)
@deprecate(SignedCholeskyPivoted{T,S}(factors, uplo, piv, rank, tol, info) where {T,S<:AbstractMatrix},
           SignedCholeskyPivoted{T,S,typeof(piv)}(factors, uplo, piv, rank, tol, info), false)


# iteration for destructuring into components
Base.iterate(F::SignedCholeskyPivoted) =  (F.uplo == 'L' ? F.L : F.U, Val(1))
Base.iterate(F::SignedCholeskyPivoted,::Val{1}) = (F.S, Val(2))
Base.iterate(F::SignedCholeskyPivoted,::Val{2}) = nothing


### Non BLAS/LAPACK element types (generic). Since generic fallback for pivoted signed Cholesky
### is not implemented yet we throw an error
signedcholesky!(A::RealHermSymComplexHerm{<:Real}, ::RowMaximum; tol = 0.0, check::Bool = true) =
    throw(ArgumentError("generic pivoted signed Cholesky factorization is not implemented yet"))
@deprecate signedcholesky!(A::RealHermSymComplexHerm{<:Real}, ::Val{true}; kwargs...) signedcholesky!(A, RowMaximum(); kwargs...) false

function getproperty(C::SignedCholeskyPivoted{T}, d::Symbol) where {T}
    Cfactors = getfield(C, :factors)
    Cuplo    = getfield(C, :uplo)
    if d === :U
        return UpperTriangular(sym_uplo(Cuplo) == d ? Cfactors : copy(Cfactors'))
    elseif d === :L
        return LowerTriangular(sym_uplo(Cuplo) == d ? Cfactors : copy(Cfactors'))
    elseif d === :p
        return getfield(C, :piv)
    elseif d === :P
        n = size(C, 1)
        P = zeros(T, n, n)
        for i = 1:n
            P[getfield(C, :piv)[i], i] = one(T)
        end
        return P
    else
        return getfield(C, d)
    end
end
Base.propertynames(F::SignedCholeskyPivoted, private::Bool=false) =
    (:U, :L, :p, :P, (private ? fieldnames(typeof(F)) : ())...)


adjoint(C::Union{SignedCholesky,SignedCholeskyPivoted}) = C


function AbstractMatrix(F::SignedCholeskyPivoted)
    ip = invperm(F.p)
    U = F.U[1:F.rank,ip]
    U'U
end
AbstractArray(F::SignedCholeskyPivoted) = AbstractMatrix(F)
Matrix(F::SignedCholeskyPivoted) = Array(AbstractArray(F))
Array(F::SignedCholeskyPivoted) = Matrix(F)

SignedCholeskyPivoted{T}(C::SignedCholeskyPivoted{T}) where {T} = C
SignedCholeskyPivoted{T}(C::SignedCholeskyPivoted) where {T} =
    SignedCholeskyPivoted(AbstractMatrix{T}(C.factors),C.uplo,C.piv,C.rank,C.tol,C.info)
Factorization{T}(C::SignedCholeskyPivoted{T}) where {T} = C
Factorization{T}(C::SignedCholeskyPivoted) where {T} = SignedCholeskyPivoted{T}(C)


copy(C::SignedCholeskyPivoted) = SignedCholeskyPivoted(copy(C.factors), C.uplo, C.piv, C.rank, C.tol, C.info)


function show(io::IO, mime::MIME{Symbol("text/plain")}, C::SignedCholeskyPivoted)
    summary(io, C); println(io)
    println(io, "$(C.uplo) factor with rank $(rank(C)):")
    show(io, mime, C.uplo == 'U' ? C.U : C.L)
    println(io, "\npermutation:")
    show(io, mime, C.p)
end

# generic computation for non-BLAS/LAPACK element types 

# sign extraction from a 2 x2 symmetric matrix 

@inline function _sign2x2(a11, a12, a22)
    # matrix = [a11 a12; a12' a22]
    tr = a11 + a22
    det = a11*a22 - abs2(a12)
    if det > 0
        if tr > 0
            return Int8(1), Int8(1)
        else
            return Int8(-1), Int8(-1)
        end
    elseif det < 0
        return Int8(1), Int8(-1)
    else
        return Int8(0), tr > 0 ? Int8(1) : (tr < 0 ? Int8(-1) : Int8(0)) 
    end
end

function _sgndchol_pivoted_generic!(A::AbstractMatrix{T}, rowmax::AbstractVector{T};
                                    tol::Real=0.0) where T<:Real
    n = size(A,1)
    require_one_based_indexing(A)
    S = Vector{Int8}(undef, n)
    p = collect(1:n)
    rank = 0

    for k = 1:n
        # pivoting
        maxindex = findmax(rowmax[k:n])[2] + k - 1
        if maxindex != k
            A[:, (k, maxindex)] .= A[:, (maxindex, k)]
            A[(k, maxindex), :] .= A[(maxindex, k), :]
            p[(k, maxindex)] .= p[(maxindex, k)]
            rowmax[(k, maxindex)] .= rowmax[(maxindex, k)]
        end
        # compute sign
        akk = A[k,k]
        for j = 1:k-1
            akk -= S[j]*A[j,k]^2
        end
        if abs(akk) <= tol
            S[k] = Int8(0)
        else
            S[k] = akk > 0 ? Int8(1) : Int8(-1)
            rank += 1
            # update trailing submatrix and rowmax
            for i = k+1:n
                aik = A[k,i]
                for j = 1:k-1
                    aik -= S[j]*A[j,k]*A[j,i]
                end
                A[k,i] = aik
                A[i,k] = aik # symmetry
                A[i,i] -= S[k]*aik^2
                rowmax[i] = maximum(abs.(A[i, k+1:end]))
            end
        end
    end
    return (S, p, rank)
end







end #module 