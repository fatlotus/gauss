package gauss

import (
	"fmt"
	"math"
)

// Array represents a dense vector, matrix, or higher-order tensor.
type Array struct {
	Data  []float64 // stored in C order
	Shape []int     // dimension, slow to fast
}

// Transpose computes the transpose of a vector or matrix.
func (a Array) Transpose() Array {
	upcast(&a, 2)
	reversed := make([]int, len(a.Shape))
	for i := range reversed {
		reversed[len(a.Shape)-1-i] = a.Shape[i]
	}
	trans := Zero(reversed...)
	for i := 0; i < a.Shape[0]; i++ {
		for j := 0; j < a.Shape[1]; j++ {
			*trans.I(j, i) = *a.I(i, j)
		}
	}
	return trans
}

// Formats the given array as an appropriately shaped list of lists.
func (a Array) String() string {
	if len(a.Shape) == 0 {
		return fmt.Sprintf("%v", a.Data[0])
	}
	result := ""
	s := a.Shape[len(a.Shape)-1]
	for i := 0; i < len(a.Data); i += s {
		for range a.Shape[1:] {
			if i == 0 {
				result += "["
			} else {
				result += " "
			}
		}
		result += "["
		for j := 0; j < s; j++ {
			if j != 0 {
				result += " "
			}
			result += fmt.Sprintf("%v", a.Data[i+j])
		}
		result += "]\n"
	}
	result = result[:len(result)-1]
	for range a.Shape[1:] {
		result += "]"
	}
	return result
}

// Zero allocates and zeros an Array with the given dimensions.
func Zero(size ...int) Array {
	prod := 1
	for _, comp := range size {
		prod *= comp
	}
	return Array{
		Data:  make([]float64, prod),
		Shape: size,
	}
}

// Returns a pointer to the given element of the matrix, checking bounds.
func (a *Array) I(comps ...int) *float64 {
	index := 0
	for i, dimension := range a.Shape {
		if comps[i] >= dimension || comps[i] < 0 {
			panic("out of range")
		}
		index = index*dimension + comps[i]
	}
	return &a.Data[index]
}

func upcast(a *Array, size int) {
	for len(a.Shape) < size {
		a.Shape = append(a.Shape, 1)
	}
}

// Adds the two arrays together compontentwise.
// It is an error to add two arrays of differing dimension.
func Sum(a, b Array) Array {
	upcast(&a, len(b.Shape))
	upcast(&b, len(b.Shape))

	if len(a.Shape) != len(b.Shape) {
		panic(fmt.Sprintf("dimension mismatch: %v + %v\n",
			a.Shape, b.Shape))
	}

	for i := range a.Shape {
		if a.Shape[i] != b.Shape[i] {
			panic(fmt.Sprintf("dimension mismatch: %v + %v\n",
				a.Shape, b.Shape))
		}
	}

	result := Array{
		Data:  make([]float64, len(a.Data)),
		Shape: a.Shape,
	}

	for i := 0; i < len(a.Data); i++ {
		result.Data[i] = a.Data[i] + b.Data[i]
	}

	return result
}

// Computes the matrix product of the two arrays. It is an error to multiply
// higher-order tensors or matrices of mismatched dimension.
func Product(a, b Array) Array {
	upcast(&a, 2)
	upcast(&b, 2)

	if len(a.Shape) != 2 || len(b.Shape) != 2 || a.Shape[1] != b.Shape[0] {
		panic(fmt.Sprintf("dimension mismatch: %v * %v\n",
			a.Shape, b.Shape))
	}

	sa := a.Shape
	sb := b.Shape

	n := sa[0]
	m := sa[1]
	p := sb[1]

	result := Zero(n, p)
	for i := 0; i < n; i++ {
		for j := 0; j < p; j++ {
			for k := 0; k < m; k++ {
				*result.I(i, j) += *a.I(i, k) * *b.I(k, j)
			}
		}
	}

	return result
}

// Returns an identity matrix of the given dimension.
func Identity(size int) Array {
	buf := Zero(size, size)
	for i := 0; i < size; i++ {
		*buf.I(i, i) = 1.0
	}
	return buf
}

// Scalar generates a zero-dimension tensor from a scalar.
func Scalar(value float64) Array {
	return Array{
		Data:  []float64{value},
		Shape: []int{},
	}
}

// Vector generates a one-dimensional tensor from a scalar.
func Vector(vals []float64) Array {
	return Array{
		Data:  vals,
		Shape: []int{len(vals)},
	}
}

// Matrix generates a two-dimensional tensor from a scalar.
func Matrix(vals [][]float64) Array {
	rows := len(vals)
	cols := len(vals[0])

	flattened := make([]float64, rows*cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			flattened[j+i*cols] = vals[i][j]
		}
	}

	return Array{
		Data:  flattened,
		Shape: []int{rows, cols},
	}
}

// Norm computes the p-norm of this tensor of the given order.
func (a Array) Norm(exp float64) float64 {
	sum := float64(0)
	for i := 0; i < len(a.Data); i++ {
		sum += math.Pow(a.Data[i], exp)
	}
	return math.Pow(sum, 1/exp)
}

// Equals computes the elementwise difference of the two tensors.
// Returns true if every corresponding element is within eps of each other.
func Equals(a, b Array, eps float64) bool {
	upcast(&a, len(b.Shape))
	upcast(&b, len(a.Shape))

	if len(a.Shape) != len(b.Shape) {
		panic(fmt.Sprintf("dimension mismatch: %v == %v\n",
			a.Shape, b.Shape))
	}

	for i := range a.Shape {
		if a.Shape[i] != b.Shape[i] {
			panic(fmt.Sprintf("dimension mismatch: %v == %v\n",
				a.Shape, b.Shape))
		}
	}

	for i := 0; i < len(a.Data); i++ {
		if math.Abs(a.Data[i]-b.Data[i]) > eps {
			return false
		}
	}

	return true
}
