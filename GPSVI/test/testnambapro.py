from numbapro import guvectorize
from numpy import zeros, arange

@guvectorize(['void(int32[:], int32[:])'], '(n)->()')
def sum_row(inp, out):
    """
    Sum every row

    function type: two arrays
                   (note: scalar is represented as an array of length 1)
    signature: n elements to scalar
    """
    tmp = 0.
    for i in range(inp.shape[0]):
        tmp += inp[i]
    out[0] = tmp

inp = arange(15, dtype='int32').reshape(5, 3)
print(inp)

# implicit output array
out = sum_row(inp)
print('imp: %s' % out)

# explicit output array
explicit_out = zeros(5, dtype='int32')
sum_row(inp, out=explicit_out)
print('exp: %s' % explicit_out)