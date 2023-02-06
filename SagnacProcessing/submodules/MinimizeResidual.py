#!/usr/bin/env python
# coding: utf-8


def __minimize_residual(model, original):

    from scipy.optimize import leastsq

    ## define cost function
    def __cost_function(params, x, y):
        a, b = params[0], params[1]
        residual = y-(a*x+b)
        return residual

    ## initials 
    params = [1,0]

    result = leastsq(__cost_function, params, (model, original))

    model_new = model * result[0][0] + result[0][1]

    print(f'\noptimized: original -  {round(result[0][0],3)} * model + {round(result[0][1],3)}\n')
    
    residual = (model_new - original)
    
    return residual, model_new