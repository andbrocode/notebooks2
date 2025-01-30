def __regression(ddf, _features, target="fj_fs", reg="theilsen", verbose=True, odr_intercept_zero=False):

    from sklearn import linear_model
    from sklearn.linear_model import LinearRegression, RANSACRegressor, HuberRegressor, TheilSenRegressor
    from numpy import array
    from scipy import odr
    from numpy import power, mean, std

    _df = ddf.copy()

    # remove time and target
    try:
        _features.remove(target)
    except:
        pass
    try:
        _features.remove("time")
    except:
        pass

    # define x data
    X = _df[_features].values.reshape(-1, len(_features))

    # define y data
    y = _df[target].values

    # multi linear regression

    # ________________________________________________________
    # using OLS
    if reg.lower() == "ols":
        ols = linear_model.LinearRegression()

        # get linear model
        model = ols.fit(X, y)

        # show results
        if verbose:
            print("R2:", model.score(X, y))
            print("X0:",  model.intercept_)
            print("Coef: ",  model.coef_)
            for _f, _c in zip(_features, model.coef_):
                print(f"{_f} : {_c}")
    # ________________________________________________________
    # using RANSAC approach
    elif reg.lower() == "ransac":

        # get linear model
        model = RANSACRegressor(random_state=1).fit(X, y)

        # show results
        if verbose:
            print("R2:", model.score(X, y))
            print("IC: ", model.estimator_.intercept_)
            print("Coef: ",  model.estimator_.coef_)
            for _f, _c in zip(_features, model.estimator_.coef_):
                print(f"{_f} : {_c}")

    # ________________________________________________________
    # using TheilSen approach
    elif reg.lower() == "theilsen":

        # get linear model
        model = TheilSenRegressor().fit(X, y)

        # show results
        if verbose:
            print("R2:", model.score(X, y))
            print("X0:",  model.intercept_)
            print("Coef: ",  model.coef_)
            for _f, _c in zip(_features, model.coef_):
                print(f"{_f} : {_c}")

    # ________________________________________________________
    # using ODR approach
    elif reg.lower() == "odr":

        # prepare errors
        xerr, yerr = std(X), std(y)

        # model for intercept = 0
        def modelx(b, x):
            return b*x + 0

        # data = odr.RealData(x_array, y_array)
        data = odr.Data(X, y, wd=1./xerr, we=1./yerr)

        # prepare output dictionary
        out = {}

        if odr_intercept_zero:
            output = odr.ODR(data, model=odr.Model(modelx), beta0=[1]).run()
            slope = output.beta
        else:
            output = odr.ODR(data, model=odr.unilinear).run()
            slope, intercept = output.beta

    # prediction
    if reg.lower() != "odr":

        model_predict = []

        for o, row in _df[_features].iterrows():

            x_pred = []
            for feat in _features:
                x_pred.append(row[feat])

            x_pred = array(x_pred)
            x_pred = x_pred.reshape(-1, len(_features))

            model_predict.append(model.predict(x_pred))

    # prepare putput dict
    out = {}

    out['model'] = model
    out['r2'] = model.score(X, y)

    # try to append prediction
    try:
        out['dp'] = model_predict
    except:
        pass

    # try to append time data
    try:
        out['tp'] = _df.time
    except:
        pass

    # append output
    if reg.lower() == "ransac":
        out['slope'] = model.estimator_.coef_
        out['inter'] = model.estimator_.intercept_
    elif reg.lower() == "theilsen":
        out['slope'] = model.coef_
        out['inter'] = model.intercept_
    elif reg.lower() == "ols":
        out['slope'] = model.coef_
        out['inter'] = model.intercept_
    elif reg.lower() == "odr":
        out['slope'] = slope
        try:
            out['inter'] = intercept
        except:
            pass

    return out