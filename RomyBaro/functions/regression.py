def __regression(ddf, _features, target="fj_fs", reg="theilsen", verbose=True):

    from sklearn import linear_model
    from sklearn.linear_model import LinearRegression, RANSACRegressor, HuberRegressor, TheilSenRegressor
    from numpy import array

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

    # using OLS
    if reg.lower() == "ols":
        ols = linear_model.LinearRegression()
        model = ols.fit(X, y)
        if verbose:
            print("R2:", model.score(X, y))
            print("X0:",  model.intercept_)
            print("Coef: ",  model.coef_)
            for _f, _c in zip(_features, model.coef_):
                print(f"{_f} : {_c}")

    elif reg.lower() == "ransac":
        model = RANSACRegressor(random_state=1).fit(X, y)
        if verbose:
            print("R2:", model.score(X, y))
            print("IC: ", model.estimator_.intercept_)
            print("Coef: ",  model.estimator_.coef_)
            for _f, _c in zip(_features, model.estimator_.coef_):
                print(f"{_f} : {_c}")

    # using TheilSen
    elif reg.lower() == "theilsen":
        model = TheilSenRegressor().fit(X, y)
        if verbose:
            print("R2:", model.score(X, y))
            print("X0:",  model.intercept_)
            print("Coef: ",  model.coef_)
            for _f, _c in zip(_features, model.coef_):
                print(f"{_f} : {_c}")

    # prediction
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
    out['tp'] = _df.time
    out['dp'] = model_predict

    if reg.lower() == "ransac":
        out['slope'] = model.estimator_.coef_
        out['inter'] = model.estimator_.intercept_
    elif reg.lower() == "theilsen":
        out['slope'] = model.coef_
        out['inter'] = model.intercept_
    elif reg.lower() == "ols":
        out['slope'] = model.coef_
        out['inter'] = model.intercept_

    # plt.plot(_df.time, model_predict)
    # plt.plot(_df.time, _df[target].values)

    return out