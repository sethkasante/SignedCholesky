
module SignedCholesky


"""
    SignedCholesky

Signed Cholesky–type factorizations for real symmetric and Hermitian matrices.

This package provides:

  • Unpivoted signed Cholesky factorizations `A ≈ L * S * Lᵀ`
  • Pivoted variants when Pivoted() is provided: `A[p,p] ≈ L * S * Lᵀ` 
  • Linear-algebraic operations derived from the factorization
    (determinant, inertia, signature, etc.)

The signed Cholesky factorization generalizes standard Cholesky
to indefinite matrices and fails if 2×2 pivots are required.
"""

# --------------
# Dependencies
# --------------
using LinearAlgebra

# abstract type 
abstract type SignedFactorization{T} <: Factorization{T} end


#LinearAlgebra functions (not imported): 
RealHmtSymComplexHmt = Union{Hermitian{T, S}, Hermitian{Complex{T}, S}, Symmetric{T, S}} where {T<:Real, S}

function _checksquare(M)
    sizeM = Base.size(M)
    length(sizeM) == 2 || throw(DimensionMismatch(lazy"input is not a matrix: dimensions are $sizeM"))
    sizeM[1] == sizeM[2] || throw(DimensionMismatch(lazy"matrix is not square: dimensions are $sizeM"))
    return sizeM[1]
end

_promote_copy(A::AbstractArray, ::Type{T}) where {T} = copyto!(similar(A, T, Base.size(A)), A)

import Base: 
        require_one_based_indexing, 
        copy, 
        show,
        permute!,
        invpermute!

#--- Include files ----

# Unpivoted signed Cholesky
include("signedcholesky_nopivot.jl")

export signedcholesky,
       signedcholesky!,
       SignedChol,
       SignedCholPivoted,
       Pivoted,
       issuccess,
       isnonfactorizable

# Pivoted signed Cholesky
include("signedcholesky_pivot.jl")

# Linear algebra utilities (det, inertia, signature, ...)
include("linalg.jl")

export
    det,
    logdet,
    logabsdet,
    inertia,
    isposdef,
    ldiv!

end # module SignedCholesky