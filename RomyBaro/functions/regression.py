def __regression(ddf, _features, target="fj_fs", reg="theilsen"):

    from sklearn import linear_model
    from sklearn.linear_model import LinearRegression, RANSACRegressor, HuberRegressor, TheilSenRegressor
    from numpy import array, power, mean, std
    from scipy import odr

    # make temporary copy
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

    print(f"using parameters: {_features}")

    # define x data
    X = _df[_features].values.reshape(-1, len(_features))

    # define y data
    y = _df[target].values

    # ________________________________________
    # multi linear regression

    # ____________
    # using ODR
    if reg.lower() == "odr":

        # set statistic measures
        # bx, by = 0, mean(y)/mean(X)
        # xerr, yerr = std(X), std(y)

        # format data
        data = odr.RealData(X, y)
        # data = odr.Data(X, y, wd=1./xerr, we=1./yerr)

        # setup regression
        odr = odr.ODR(data, model=odr.unilinear)

        # compute regression
        output = odr.run()

        odr_slope, odr_intercept = output.beta

        model = []

    # ____________
    # using OLS
    elif reg.lower() == "ols":
        ols = linear_model.LinearRegression()
        model = ols.fit(X, y)
        r2 = model.score(X, y)
        print("R2:", model.score(X, y))
        print("X0:",  model.intercept_)
        print("Coef: ",  model.coef_)
        for _f, _c in zip(_features, model.coef_):
            print(f"{_f} : {_c}")

    # ____________
    # using Ransac
    elif reg.lower() == "ransac":
        model = RANSACRegressor(random_state=1).fit(X, y)
        r2 = model.score(X, y)

        print("R2:", model.score(X, y))
        print("IC: ", model.estimator_.intercept_)
        print("Coef: ",  model.estimator_.coef_)
        for _f, _c in zip(_features, model.estimator_.coef_):
            print(f"{_f} : {_c}")

    # ____________
    # using TheilSen
    elif reg.lower() == "theilsen":
        model = TheilSenRegressor().fit(X, y)
        r2 = model.score(X, y)

        print("R2:", model.score(X, y))
        print("X0:",  model.intercept_)
        print("Coef: ",  model.coef_)
        for _f, _c in zip(_features, model.coef_):
            print(f"{_f} : {_c}")


    # ________________________________________
    # prediction
    model_predict = []
    if reg.lower() != "odr":

        for o, row in _df[_features].iterrows():

            x_pred = []
            for feat in _features:
                x_pred.append(row[feat])

            x_pred = array(x_pred)
            x_pred = x_pred.reshape(-1, len(_features))

            model_predict.append(model.predict(x_pred))

    # ________________________________________
    # prepare output dict
    out = {}
    out['model'] = model
    out['r2'] = r2
    out['tp'] = _df.time
    out['dp'] = model_predict

    # plt.plot(_df.time, model_predict)
    # plt.plot(_df.time, _df[target].values)

    return out