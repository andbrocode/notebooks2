import numpy as np

def masking(img, n=3, method='mean', stats=[None, None]):
    '''
    Search for outlayer pixels with a value lower/larger
    than the mean/median/other ± n × std
    '''
    if method.lower() == 'average':
        avg, std = stats[0], stats[1]
    elif method == 'mean':
        avg, std = np.nanmean(img), np.nanstd(img)
    elif method == 'median':
        avg, std = np.nanmedian(img), np.nanstd(img)
    return (img < (avg - n * std)) | (img > (avg + n * std))

def box(A, i, j, size=1, nan=False):
    '''
    Extract the box (subarray) around (i,j) value inside
    a box of size `size` and replace the centrale
    value by NaN if necessary [option].
    '''
    if not isinstance(A, (list, np.ndarray)):
        raise TypeError(
            'The input must be an array. {} provided'.format(type(A)))

    ny, nx = A.shape
    if i < 0:
        raise ValueError('Line index `i` must be ≥ 0')
    if j < 0:
        raise ValueError('Column index `j` must be ≥ 0')
    if i >= nx:
        raise ValueError('Line index `i` must be < {}'.format(nx))
    if j >= ny:
        raise ValueError('Column index `j` must be < {}'.format(ny))

    if not isinstance(size, int):
        raise TypeError('Size must be an int')
    if size < 1:
        raise ValueError('The box size must be ≥ 1')

    if nan:
        # The change of type is required to store a NaN
        A = np.array(A, dtype=np.double)
        A[j, i] = np.nan

    l = i - size if i - size > 0 else 0
    r = i + size if i + size < nx else nx
    t = j - size if j - size > 0 else 0
    b = j + size if j + size < ny else ny

    return A[t:b+1, l:r+1]

def mean(A):
    return np.nanmean(A), np.nanstd(A)

def median(A):
    return np.nanmedian(A), np.nanstd(A)

def spikes(img, method='mean', size=2, n=3, thres=None):
    '''
    Search spikes in the image using a moving box
    of size `size` using `method` method with ± n × std
    '''
    if method.lower() == 'mean':
        mask = mean
    elif method.lower() == 'median':
        mask = median
    elif method.lower() == 'average':
        mask = None
        m, s = np.median(img), 1e-8
    else:
        raise ValueError("The masking method must be '[mean|median]")

    # s = np.nanpercentile(abs(img), 98)

    ny, nx = img.shape
    _spikes = np.zeros((ny, nx))
    for i in range(nx):
        for j in range(ny):
            m, s = mask(box(img, i, j, size, nan=True))

            # set threshold
            if thres is not None:
                if n * s < thres:
                    s = thres

            if img[j, i] < m - n * s or img[j, i] > m + n * s:
                _spikes[j, i] = 1

    return _spikes == 1

def fill(img, mask, method='mean', size=1):
    '''
    Fill the masked pixels with the mean value of
    surrounding neighboors inside a box of size `size`
    '''
    if img.shape != mask.shape:
        raise ValueError('Image and outlayers mask must have the same dimentsion: {} vs. {}'.format(
            img.shape, mask.shape
        ))

    if method.lower() == 'mean':
        f = np.nanmean
    elif method.lower() == 'median':
        f = np.nanmedian
    elif method.lower() == 'nan':
        f = np.nanmedian
    else:
        raise ValueError("The filling method must be '[mean|median]")

    A = np.copy(img)
    ny, nx = A.shape
    x, y = np.meshgrid(range(nx), range(ny))

    for x, y in zip(x[mask], y[mask]):
        A[y, x] = f(box(A, x, y, size=size, nan=True))
        # if method.lower() == 'nan':
        #     A *= np.nan
        #     print(A, type(A))
    return A


def clean(img, mask='mean', size=2, n=3, fill_method='median', fill_size=1, thres=None):
    '''
    Clean image from spikes with the `fill_method` method with surrounding
    neighboors inside a box of size `fill_size`.
    '''

    # use absolute of image (seismic data)
    aimg = abs(img)

    return fill(img, spikes(aimg, mask, size, n, thres), fill_method, fill_size), spikes(aimg, mask, size, n, thres)

def signal_to_image(arr):
    '''
    1d signal to square image
    '''
    # estimate size of square image
    N = int(np.ceil(np.sqrt(len(arr))))

    # compute difference required for square image
    Npad = N**2 - len(arr)

    # pad additional values with last value of signal
    _pad = np.append(arr, np.ones(Npad)*arr[-1])

    # reshape to square image
    img = _pad.reshape(N, N)

    return img, Npad

def image_to_signal(img, pad=0):
    '''
    reshape image to 1d signal
    '''
    return img.reshape(1, img.size)[0][:-pad]


def get_mask(spikes):
    '''
    modify 1d spikes to mask
    '''
    mask = np.where(spikes > 0, -1, spikes)
    mask = np.where(mask == 0, 1, mask)
    mask = np.where(mask < 0, np.nan, mask)
    return mask

def smooth(y, npts):
    '''
    moving average of 1d signal for n samples
    '''
    win = np.hanning(npts)
    y_smooth = np.convolve(y, win/np.sum(win), mode='same')
    y_smooth[:npts//2] = np.nan
    y_smooth[-npts//2:] = np.nan
    return y_smooth
