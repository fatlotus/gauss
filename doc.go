/*
Package gauss is a dense numerical linear algebra library for Go.

It is intended to allow fast, easy access to matrices, vectors, and scalars,
as well as a few basic algorithms to make larger systems easier. A typical use
might be computing the principal components of a matrix of data using the
singular value decomposition.

Basics

Everything in Gauss is represented by a gauss.Array. (You can think of it
as being somewhat analagous to a numpy.array.) Like a slice, an Array has
underlying data, stored as double-precision floats, and a shape, determining
the axes that are used for the data.

To create an Array, use one of the helper functions:

	a := Vector([]float64{1, 2, 3})
	m := Matrix([][]float64{
		{1, 2},
		{3, 4},
	})

This sets up the buffers automatically. There are also several functions to aid
in common types of matrices:

	i := Identity(4) // an identity matrix of dim 4
	z := Zero(3) // a zero-vector of length 3
	d := Diagonal(a) // a diagonal matrix
	r := Random(5) // random 5x5 matrix from [0, 1)

Once you've created several Arrays, you can do elementary mathematical
operations on them:

	c := Product(m, a)
	d := Sum(z, c)
	e := m.Transpose()
	g := m.Inverse()
	h := m.Scale(-1)

Each of the above allocates a new buffer to store the result.

Algorithms

As of this writing, we only have one algorithm. That is the singular value
decomposition algorithm borrowed from the Colt project. Using it is fairly
simple. In the below, we are performing rank reduction on m:

	u, s, v := SVD(m)
	for i := 1; i < len(s.Data); i++ {
		s.Data[i] = 0
	}
	f := Diagonal(s)
	mp := Product(u, Product(f, v.Transpose()))

Since the SVD is something of a workhorse algorithm, it tends to be fairly
useful across numerical computing.
*/
package gauss
