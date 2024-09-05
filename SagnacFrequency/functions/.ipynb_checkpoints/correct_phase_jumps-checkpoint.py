def __correct_phase_jumps(data, detect, di=10):

    import numpy as np

    data = np.array(data)

    # offset from jump
    offset = 3

    for _i in range(len(data)):

        # avoid negative i
        if _i < di+offset:
            continue

        if detect[_i] == 1:
            try:
                left = np.nanmean(data[_i-offset-di:_i-offset])
                right = np.nanmean(data[_i+offset:_i+offset+di])
            except:
                print(f" -> skip {_i}")
                continue

            # determine difference
            diff = left - right

            # adjust for jump
            data[_i-3:_i] = left
            data[_i:] += diff

    return data