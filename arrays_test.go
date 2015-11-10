package gauss

import (
	"fmt"
	"math"
	"testing"
)

// An Array is just a slice annotated with a dimension. In this example, we've
// demonstrated several ways to create Arrays: one using raw literal notation,
// and another using a helper function.
func ExampleArray() {
	// Construct a matrix using literal notation.
	a := Array{
		Data:  []float64{1, 0, 0, 1},
		Shape: []int{2, 2},
	}
	fmt.Printf("a:\n%v\n", a)

	// Construct another matrix using a helper.
	a = Matrix([][]float64{
		{1, 2},
		{3, 4},
	})
	fmt.Printf("a2:\n%v\n", a)

	// Reshape the array to turn the matrix into a vector.
	a.Shape = []int{4}

	fmt.Printf("a3:\n%v\n", a)
	// Output:
	// a:
	// [[1 0]
	//  [0 1]]
	// a2:
	// [[1 2]
	//  [3 4]]
	// a3:
	// [1 2 3 4]
}

// The Equals function is useful for comparing matrices in the presence of
// rounding errors. In this example, two matrices that are mostly the same
// are still "Equals" each other, despite being slightly distinct.
func ExampleEquals() {
	// Generate a number slightly greater than 2.
	twoPlusAHair := math.Nextafter(2, 3)
	a := Vector([]float64{2, 3, 4})
	ap := Vector([]float64{twoPlusAHair, 3, 4})

	// Check that these values are equal.
	fmt.Printf("(%v =?= %v) = %v\n", a, a, Equals(a, a, 0))
	fmt.Printf("(%v =?= %v) = %v\n", a, ap, Equals(a, a, 0.0001))
	// Output:
	// ([2 3 4] =?= [2 3 4]) = true
	// ([2 3 4] =?= [2.0000000000000004 3 4]) = true
}

func TestEqual(t *testing.T) {
	a := Vector([]float64{2, 3, 4})
	if !Equals(a, a, 0) {
		t.Fatal("a != a")
	}

	b := Vector([]float64{3, 4, 5})
	if Equals(a, b, 0) {
		t.Fatal("a == b")
	}
}

// Compute the product of a vector and a matrix. If a vector is
// one-dimensional, it is first upcasted to a column vector before being
// multiplied.
func ExampleProduct() {
	// Compute the matrix product.
	a := Vector([]float64{2, 3, 4})
	m := Matrix([][]float64{
		{1, 2, 3},
		{4, 5, 6},
		{7, 8, 9},
	})
	fmt.Printf("m * a:\n%v\n", Product(m, a))

	// Compute the dot product of two vectors.
	b := Vector([]float64{4, 5, 6})
	fmt.Printf("at * b:\n%v\n", Product(a.Transpose(), b))
	// Output:
	// m * a:
	// [[20]
	//  [47]
	//  [74]]
	// at * b:
	// [[47]]
}

// The Identity matrix is a matrix that, when multiplied with any vector,
// produces the same vector. It can be built by creating a matrix with ones
// along the diagonal and zeros elsewhere.
func ExampleIdentity() {
	i := Identity(4)
	fmt.Printf("%v\n", i)

	// Left multiplication keeps the vector the same.
	b := Vector([]float64{1, 2, 3, 4})
	fmt.Printf("I * b:\n%v\n", Product(i, b))

	// Right multiplication keeps the vector the same.
	fmt.Printf("bt * I:\n%v\n", Product(b.Transpose(), i))
	// Output:
	// [[1 0 0 0]
	//  [0 1 0 0]
	//  [0 0 1 0]
	//  [0 0 0 1]]
	// I * b:
	// [[1]
	//  [2]
	//  [3]
	//  [4]]
	// bt * I:
	// [[1 2 3 4]]
}

// Addition works the way you'd expect it to: by adding corresponding parts of
// each vector.
func ExampleSum() {
	// Add to vectors component-wise.
	a := Vector([]float64{2, 3, 4})
	b := Vector([]float64{4, 5, 6})
	fmt.Printf("a + b: %v\n", Sum(a, b))
	// Output: a + b: [6 8 10]
}

// The 2-norm of a matrix in Eucliean space is equivalent to the Pythagorean
// theorem (since the basis elements are orthogonal). Hence we can use Norm to
// solve for the hypotenuse of a right triangle:
func ExampleArray_Norm() {
	// Compute the third side of a 3-4-5 triangle.
	a := Vector([]float64{3, 4})
	fmt.Printf("length of hypotenuse: %v\n", a.Norm(2))
	// Output: length of hypotenuse: 5
}

func ExampleArray_Transpose() {
	a := Matrix([][]float64{
		{1, 2, 3},
		{4, 5, 6},
	})
	aT := a.Transpose()
	fmt.Printf("aT:\n%v\n", aT)
	// Output:
	// aT:
	// [[1 4]
	//  [2 5]
	//  [3 6]]
}

// The I helper function allows easy access to the elements of an Array.
func ExampleArray_I() {
	v := Matrix([][]float64{
		{1, 2},
		{3, 4},
	})

	// Get a cell in the matrix.
	fmt.Printf("%v\n", *v.I(1, 1))

	// Update a cell in the matrix.
	*v.I(0, 1) = 10
	fmt.Printf("%v\n", v)
	// Output:
	// 4
	// [[1 10]
	//  [3 4]]
}

func ExampleArray_String() {
	m := Matrix([][]float64{
		{1, 2, 3},
		{4, 5, 6},
	})
	v := Vector([]float64{
		7, 8, 9,
	})
	s := Scalar(10)
	fmt.Printf("%v\n%v\n%v\n", m, v, s)
	// Output:
	// [[1 2 3]
	//  [4 5 6]]
	// [7 8 9]
	// 10
}
