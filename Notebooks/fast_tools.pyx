cpdef float col_0_sum(float[:, :] mat_a):
    cdef:
        int i
        float col_sum = 0.0

    for i in range(mat_a.shape[0]):
        col_sum += mat_a[i, 0]

    return col_sum
    
