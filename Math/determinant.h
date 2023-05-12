#pragma once

#define N 10
// Borrowed from https://www.geeksforgeeks.org/c-program-to-find-determinant-of-a-matrix/ on 09-05-23
void getCofactor(double mat[N][N], double temp[N][N], int p, int q, int n)
{
    int i = 0, j = 0;

    // Looping for each element of the matrix
    for (int row = 0; row < n; row++)
    {
        for (int col = 0; col < n; col++)
        {
            // Copying into temporary matrix
            // only those element which are
            // not in given row and column
            if (row != p && col != q)
            {
                temp[i][j++] = mat[row][col];

                // Row is filled, so increase row
                // index and reset col index
                if (j == n - 1)
                {
                    j = 0;
                    i++;
                }
            }
        }
    }
}

/* Recursive function for finding the
   determinant of matrix. n is current
   dimension of mat[][]. */
// Borrowed from https://www.geeksforgeeks.org/c-program-to-find-determinant-of-a-matrix/ on 09-05-23
double determinantOfMatrix(double mat[N][N], int n)
{
    // Initialize result
    double D = 0;

    //  Base case : if matrix contains
    // single element
    if (n == 1)
        return mat[0][0];

    // To store cofactors
    double temp[N][N];

    // To store sign multiplier
    int sign = 1;

    // Iterate for each element of
    // first row
    for (int f = 0; f < n; f++)
    {
        // Getting Cofactor of mat[0][f]
        getCofactor(mat, temp, 0, f, n);
        D += sign * mat[0][f] * determinantOfMatrix(temp, n - 1);

        // Terms are to be added with alternate sign
        sign = -sign;
    }

    return D;
}
