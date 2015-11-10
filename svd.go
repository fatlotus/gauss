package gauss

import (
	"fmt"
	"math"
)

// Diagonal generates a diagonal matrix with the given diagonal entries.
func Diagonal(diag []float64) Array {
	mat := Zero(len(diag), len(diag))
	for i := 0; i < len(diag); i++ {
		*mat.I(i, i) = diag[i]
	}
	return mat
}

func hypot(a, b float64) float64 {
	var r float64
	if math.Abs(a) > math.Abs(b) {
		r = b / a
		r = math.Abs(a) * math.Sqrt(1+r*r)
	} else if b != 0 {
		r = a / b
		r = math.Abs(b) * math.Sqrt(1+r*r)
	} else {
		r = 0.0
	}
	if math.Abs(r*r-b*b-a*a) > 0.001 {
		panic("hypot is wrong")
	}
	return r
}

func imin(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func imax(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// SVD computes the singular value decomposition of the given matrix A.
// It returns orthogonal tensors U, V, and vector S, where A = U Diagonal(S) V*.
func SVD(A Array) (Array, Array, Array) {
	upcast(&A, 2)
	if len(A.Shape) > 2 {
		panic(fmt.Sprintf("cannot take SVD of array with shape %v", A.Shape))
	}

	if A.Shape[0] < A.Shape[1] {
		panic(fmt.Sprintf("cannot take SVD of array with shape %v", A.Shape))
	}

	// Derived from LINPACK code.
	// Initialize.
	m := A.Shape[0]
	n := A.Shape[1]

	newdata := make([]float64, len(A.Data))
	copy(newdata, A.Data)
	A.Data = newdata

	nu := imin(m, n)
	s := Zero(imin(m+1, n))
	U := Zero(m, nu)
	V := Zero(n, n)

	e := make([]float64, n)
	work := make([]float64, m)

	wantu := true
	wantv := true

	// Reduce A to bidiagonal form, storing the diagonal elements
	// in s and the super-diagonal elements in e.

	nct := imin(m-1, n)
	nrt := imax(0, imin(n-2, m))
	for k := 0; k < imax(nct, nrt); k++ {
		if k < nct {

			// Compute the transformation for the k-th column and
			// place the k-th diagonal in s.Data[k].
			// Compute 2-norm of k-th column without under/overflow.
			s.Data[k] = 0
			for i := k; i < m; i++ {
				s.Data[k] = hypot(s.Data[k], A.Data[i*n+k])
			}
			if s.Data[k] != 0.0 {
				if A.Data[k*n+k] < 0.0 {
					s.Data[k] = -s.Data[k]
				}
				for i := k; i < m; i++ {
					A.Data[i*n+k] /= s.Data[k]
				}
				A.Data[k*n+k] += 1.0
			}
			s.Data[k] = -s.Data[k]
		}
		for j := k + 1; j < n; j++ {
			if k < nct && s.Data[k] != 0.0 {

				// Apply the transformation.

				t := 0.0
				for i := k; i < m; i++ {
					t += A.Data[i*n+k] * A.Data[i*n+j]
				}
				t = -t / A.Data[k*n+k]
				for i := k; i < m; i++ {
					A.Data[i*n+j] += t * A.Data[i*n+k]
				}
			}

			// Place the k-th row of A into e for the
			// subsequent calculation of the row transformation.

			e[j] = A.Data[k*n+j]
		}
		if wantu && k < nct {

			// Place the transformation in U for subsequent back
			// multiplication.

			for i := k; i < m; i++ {
				U.Data[i*nu+k] = A.Data[i*n+k]
			}
		}
		if k < nrt {

			// Compute the k-th row transformation and place the
			// k-th super-diagonal in e[k].
			// Compute 2-norm without under/overflow.
			e[k] = 0
			for i := k + 1; i < n; i++ {
				e[k] = hypot(e[k], e[i])
			}
			if e[k] != 0.0 {
				if e[k+1] < 0.0 {
					e[k] = -e[k]
				}
				for i := k + 1; i < n; i++ {
					e[i] /= e[k]
				}
				e[k+1] += 1.0
			}
			e[k] = -e[k]
			if k+1 < m && e[k] != 0.0 {

				// Apply the transformation.

				for i := k + 1; i < m; i++ {
					work[i] = 0.0
				}
				for j := k + 1; j < n; j++ {
					for i := k + 1; i < m; i++ {
						work[i] += e[j] * A.Data[i*n+j]
					}
				}
				for j := k + 1; j < n; j++ {
					t := -e[j] / e[k+1]
					for i := k + 1; i < m; i++ {
						A.Data[i*n+j] += t * work[i]
					}
				}
			}
			if wantv {

				// Place the transformation in V for subsequent
				// back multiplication.

				for i := k + 1; i < n; i++ {
					V.Data[i*n+k] = e[i]
				}
			}
		}
	}

	// Set up the final bidiagonal matrix or order p.

	p := imin(n, m+1)
	if nct < n {
		s.Data[nct] = A.Data[nct*n+nct]
	}
	if m < p {
		s.Data[p-1] = 0.0
	}
	if nrt+1 < p {
		e[nrt] = A.Data[nrt*n+(p-1)]
	}
	e[p-1] = 0.0

	// If required, generate U.

	if wantu {
		for j := nct; j < nu; j++ {
			for i := 0; i < m; i++ {
				U.Data[i*nu+j] = 0.0
			}
			U.Data[j*nu+j] = 1.0
		}
		for k := nct - 1; k >= 0; k-- {
			if s.Data[k] != 0.0 {
				for j := k + 1; j < nu; j++ {
					t := 0.0
					for i := k; i < m; i++ {
						t += U.Data[i*nu+k] * U.Data[i*nu+j]
					}
					t = -t / U.Data[k*nu+k]
					for i := k; i < m; i++ {
						U.Data[i*nu+j] += t * U.Data[i*nu+k]
					}
				}
				for i := k; i < m; i++ {
					U.Data[i*nu+k] = -U.Data[i*nu+k]
				}
				U.Data[k*nu+k] = 1.0 + U.Data[k*nu+k]
				for i := 0; i < k-1; i++ {
					U.Data[i*nu+k] = 0.0
				}
			} else {
				for i := 0; i < m; i++ {
					U.Data[i*nu+k] = 0.0
				}
				U.Data[k*nu+k] = 1.0
			}
		}
	}

	// If required, generate V.

	if wantv {
		for k := n - 1; k >= 0; k-- {
			if k < nrt && e[k] != 0.0 {
				for j := k + 1; j < nu; j++ {
					t := 0.0
					for i := k + 1; i < n; i++ {
						t += V.Data[i*n+k] * V.Data[i*n+j]
					}
					t = -t / V.Data[(k+1)*n+k]
					for i := k + 1; i < n; i++ {
						V.Data[i*n+j] += t * V.Data[i*n+k]
					}
				}
			}
			for i := 0; i < n; i++ {
				V.Data[i*n+k] = 0.0
			}
			V.Data[k*n+k] = 1.0
		}
	}

	// Main iteration loop for the singular values.

	pp := p - 1
	iter := 0
	eps := math.Pow(2.0, -52.0)
	for p > 0 {
		var k int
		var kase int

		// Here is where a test for too many iterations would go.

		// This section of the program inspects for
		// negligible elements in the s and e arrays.  On
		// completion the variables kase and k are set as follows.

		// kase = 1     if s(p) and e[k-1] are negligible and k<p
		// kase = 2     if s(k) is negligible and k<p
		// kase = 3     if e[k-1] is negligible, k<p, and
		//              s(k), ..., s(p) are not negligible (qr step).
		// kase = 4     if e(p-1) is negligible (convergence).

		for k = p - 2; k >= -1; k-- {
			if k == -1 {
				break
			}
			if math.Abs(e[k]) <= eps*(math.Abs(s.Data[k])+math.Abs(s.Data[k+1])) {
				e[k] = 0.0
				break
			}
		}
		if k == p-2 {
			kase = 4
		} else {
			var ks int
			for ks = p - 1; ks >= k; ks-- {
				if ks == k {
					break
				}
				t := 0.0
				if ks != p {
					t += math.Abs(e[ks])
				}
				if ks != k+1 {
					t += math.Abs(e[ks-1])
				}
				if math.Abs(s.Data[ks]) <= eps*t {
					s.Data[ks] = 0.0
					break
				}
			}
			if ks == k {
				kase = 3
			} else if ks == p-1 {
				kase = 1
			} else {
				kase = 2
				k = ks
			}
		}
		k++

		// Perform the task indicated by kase.

		switch kase {

		// Deflate negligible s(p).

		case 1:
			{
				f := e[p-2]
				e[p-2] = 0.0
				for j := p - 2; j >= k; j-- {
					t := hypot(s.Data[j], f)
					cs := s.Data[j] / t
					sn := f / t
					s.Data[j] = t
					if j != k {
						f = -sn * e[j-1]
						e[j-1] = cs * e[j-1]
					}
					if wantv {
						for i := 0; i < n; i++ {
							t = cs*V.Data[i*n+j] + sn*V.Data[i*n+(p-1)]
							V.Data[i*n+(p-1)] = -sn*V.Data[i*n+j] + cs*V.Data[i*n+(p-1)]
							V.Data[i*n+j] = t
						}
					}
				}
			}
			break

		// Split at negligible s(k).

		case 2:
			{
				f := e[k-1]
				e[k-1] = 0.0
				for j := k; j < p; j++ {
					t := hypot(s.Data[j], f)
					cs := s.Data[j] / t
					sn := f / t
					s.Data[j] = t
					f = -sn * e[j]
					e[j] = cs * e[j]
					if wantu {
						for i := 0; i < m; i++ {
							t = cs*U.Data[i*nu+j] + sn*U.Data[i*nu+(k-1)]
							U.Data[i*nu+(k-1)] = -sn*U.Data[i*nu+j] + cs*U.Data[i*nu+(k-1)]
							U.Data[i*nu+j] = t
						}
					}
				}
			}
			break

		// Perform one qr step.

		case 3:
			{

				// Calculate the shift.

				scale := math.Max(math.Max(math.Max(math.Max(
					math.Abs(s.Data[p-1]), math.Abs(s.Data[p-2])), math.Abs(e[p-2])),
					math.Abs(s.Data[k])), math.Abs(e[k]))
				sp := s.Data[p-1] / scale
				spm1 := s.Data[p-2] / scale
				epm1 := e[p-2] / scale
				sk := s.Data[k] / scale
				ek := e[k] / scale
				b := ((spm1+sp)*(spm1-sp) + epm1*epm1) / 2.0
				c := sp * epm1 * (sp * epm1)
				shift := 0.0
				if b != 0.0 || c != 0.0 {
					shift = math.Sqrt(b*b + c)
					if b < 0.0 {
						shift = -shift
					}
					shift = c / (b + shift)
				}
				f := (sk+sp)*(sk-sp) + shift
				g := sk * ek

				// Chase zeros.

				for j := k; j < p-1; j++ {
					t := hypot(f, g)
					cs := f / t
					sn := g / t
					if j != k {
						e[j-1] = t
					}
					f = cs*s.Data[j] + sn*e[j]
					e[j] = cs*e[j] - sn*s.Data[j]
					g = sn * s.Data[j+1]
					s.Data[j+1] = cs * s.Data[j+1]
					if wantv {
						for i := 0; i < n; i++ {
							t = cs*V.Data[i*n+j] + sn*V.Data[i*n+(j+1)]
							V.Data[i*n+(j+1)] = -sn*V.Data[i*n+j] + cs*V.Data[i*n+(j+1)]
							V.Data[i*n+j] = t
						}
					}
					t = hypot(f, g)
					cs = f / t
					sn = g / t
					s.Data[j] = t
					f = cs*e[j] + sn*s.Data[j+1]
					s.Data[j+1] = -sn*e[j] + cs*s.Data[j+1]
					g = sn * e[j+1]
					e[j+1] = cs * e[j+1]
					if wantu && j < m-1 {
						for i := 0; i < m; i++ {
							t = cs*U.Data[i*nu+j] + sn*U.Data[i*nu+(j+1)]
							U.Data[i*nu+(j+1)] = -sn*U.Data[i*nu+j] + cs*U.Data[i*nu+(j+1)]
							U.Data[i*nu+j] = t
						}
					}
				}
				e[p-2] = f
				iter = iter + 1
			}
			break

		// Convergence.

		case 4:
			{

				// Make the singular values positive.

				if s.Data[k] <= 0.0 {
					if s.Data[k] < 0 {
						s.Data[k] = -s.Data[k]
					} else {
						s.Data[k] = 0.0
					}
					if wantv {
						for i := 0; i <= pp; i++ {
							V.Data[i*n+k] = -V.Data[i*n+k]
						}
					}
				}

				// Order the singular values.

				for k < pp {
					if s.Data[k] >= s.Data[k+1] {
						break
					}
					t := s.Data[k]
					s.Data[k] = s.Data[k+1]
					s.Data[k+1] = t
					if wantv && k < n-1 {
						for i := 0; i < n; i++ {
							t = V.Data[i*n+(k+1)]
							V.Data[i*n+(k+1)] = V.Data[i*n+k]
							V.Data[i*n+k] = t
						}
					}
					if wantu && k < m-1 {
						for i := 0; i < m; i++ {
							t = U.Data[i*nu+(k+1)]
							U.Data[i*nu+(k+1)] = U.Data[i*nu+k]
							U.Data[i*nu+k] = t
						}
					}
					k++
				}
				iter = 0
				p--
			}
			break
		}
	}

	Up := Zero(m, imin(m+1, n))
	for i := 0; i < m; i++ {
		for j := 0; j < nu; j++ {
			*Up.I(i, j) = *U.I(i, j)
		}
	}

	return Up, s, V
}
