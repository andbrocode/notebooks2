def __correct_phase_jumps(data, detect):

    from numpy import nanmean, array

    data = array(data)

    for _i in range(len(data)):

        if detect[_i] == 1:
            left = nanmean(data[_i-8:_i-3])
            right = nanmean(data[_i+3:_i+8])

            diff = left - right

            data[_i-3:_i] = left

            data[_i:] += diff

    return data