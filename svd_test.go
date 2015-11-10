package gauss

import (
	"fmt"
	"testing"
)

// A diagonal matrix is one where all the entries A(ii) are set to the values
// of the given matrix. In this example, we construct a diagonal matrix of rank
// three.
func ExampleDiagonal() {
	a := Diagonal([]float64{1, 2, 3, 0})
	fmt.Printf("%v\n", a)
	// Output:
	// [[1 0 0 0]
	//  [0 2 0 0]
	//  [0 0 3 0]
	//  [0 0 0 0]]
}

// The inverse of a matrix is the matrix that, when multiplied, creates the
// identity.
func ExampleInverse() {
	// Invert a matrix.
	A := Matrix([][]float64{
		{3, 4},
		{5, 6},
	})
	AAi := Product(A, A.Inverse())
	fmt.Printf("A Ai = I? %v\n", Equals(AAi, Identity(2), 0.0001))
	// Output: A Ai = I? true
}

// Here we use the singular value decomposition of a matrix to invert it.
// Because A = U S Vt, where U and Vt are both orthogonal and S is diagonal,
// is composed of easily invertible matrices, the product is easy to compute.
// Hence Ai = V Si Ut.
func ExampleSVD() {
	A := Matrix([][]float64{
		{249, 66, 68},
		{104, 214, 108},
		{144, 146, 293},
	})

	U, s, V := SVD(A)

	// Compute the inverse of the singular value matrix.
	for i := range s.Data {
		s.Data[i] = 1 / s.Data[i]
	}
	S := Diagonal(s.Data)

	Ut := U.Transpose()

	// Compute the inverse of A.
	Ai := Product(V, Product(S, Ut))
	AiA := Product(Ai, A)

	// Verify that we've actually found an inverse.
	fmt.Printf("is A = Ai? %v\n", Equals(AiA, Identity(3), 0.0001))
	// Output: is A = Ai? true
}

func TestSVD(t *testing.T) {
	A := Matrix([][]float64{
		{249, 66, 68},
		{104, 214, 108},
		{144, 146, 293},
	})

	U, s, V := SVD(A)

	UUt := Product(U, U.Transpose())
	if !Equals(UUt, Identity(3), 0.000000001) {
		t.Fatalf("V * Vt =\n%v\n!= I", UUt)
	}

	VVt := Product(V, V.Transpose())
	if !Equals(VVt, Identity(3), 0.000000001) {
		t.Fatalf("V * Vt =\n%v\n!= I", VVt)
	}

	S := Diagonal(s.Data)
	Ar := Product(U, Product(S, V.Transpose()))

	if !Equals(A, Ar, 0.000000001) {
		t.Fatalf("A =\n%v\n!=\n%v\n= Ar", A, Ar)
	}
}
