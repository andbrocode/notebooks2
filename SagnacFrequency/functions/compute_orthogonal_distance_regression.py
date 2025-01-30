def __compute_orthogonal_distance_regression(x_array, y_array, intercept_zero=False, xerr=None, yerr=None, bx=None, by=None):

    from scipy import odr
    from numpy import power, mean, std

    if bx is None and by is None:
        bx, by = 0, mean(y_array)/mean(x_array)

    if xerr is None and yerr is None:
        xerr, yerr = std(x_array), std(y_array)

    def modelx(B, x):
        return B*x + 0

    # data = odr.RealData(x_array, y_array)
    data = odr.Data(x_array, y_array, wd=1./xerr, we=1./yerr)

    # prepare output dictionary
    out = {}

    if intercept_zero:
        output = odr.ODR(data, model=odr.Model(modelx), beta0=[1.]).run()
        out['slope'] = output.beta
    else:
        output = odr.ODR(data, model=odr.unilinear).run()
        out['slope'], out['intercept'] = output.beta

    return out
