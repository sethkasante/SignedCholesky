# SignedCholesky.jl #

A Julia package implementing a **Signed Cholesky factorization** for real symmetric and complex Hermitian matrices.

Signed Cholesky generalizes the standard Cholesky factorization to **indefinite but factorizable matrices**, producing a decomposition of the form 
\[ A \approx L \, S \, L^{\mathsf T} \quad\text{or} \quad A \approx U^{\mathsf T} \, S \, U \]
