
import os, errno
import numpy as np

def mkdir_p(path):
    """
    Create directory given path.
    """
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
        
def integrate_2d(deltas, func):
    """
    2D Integral numeric approximation.
    """

    suma = 0
    for i in range(len(func)-1):
        for j in range(len(func[0])-1):
            suma += func[i][j] + func[i+1][j] + func[i][j+1] + func[i+1][j+1]
            
    step = 1.
    for d in deltas:
        step *= d

    return 0.25*suma*step


def cartesian(arrays, out=None):
    """
    Compute cartesian product between vector set.
    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = int(n / arrays[0].size)
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in range(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out




def opt_L_curve(xs, ys):
    """
    Generic approach to find optimal Y for L-curve.
    """

    x0, y0 = xs[0], ys[0]
    x1, y1 = xs[-1], ys[-1]
    ra = float(y0 - y1) / (x0 - x1)
    rb = y1 - ra*x1
    result = []
    for xp, yp in zip(xs, ys):
        da = -1./ra
        db = yp - da * xp
        x_star = float(db-rb)/(ra-da)
        y_star = ra*x_star + rb
        result.append( [np.sqrt((xp-x_star)**2 + (yp-y_star)**2), xp, yp] )

    return max(result, key=lambda x: x[0])[2]
