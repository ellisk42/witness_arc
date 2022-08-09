import numpy as np
import numpy
import numbers
#from skimage.morphology import flood_fill
from skimage.morphology._flood_fill_cy import _flood_fill_equal, _flood_fill_tolerance

def _set_border_values(image, value, border_width=1):
    """Set edge values along all axes to a constant value.
    Parameters
    ----------
    image : ndarray
        The array to modify inplace.
    value : scalar
        The value to use. Should be compatible with `image`'s dtype.
    border_width : int or sequence of tuples
        A sequence with one 2-tuple per axis where the first and second values
        are the width of the border at the start and end of the axis,
        respectively. If an int is provided, a uniform border width along all
        axes is used.
    Examples
    --------
    >>> image = np.zeros((4, 5), dtype=int)
    >>> _set_border_values(image, 1)
    >>> image
    array([[1, 1, 1, 1, 1],
           [1, 0, 0, 0, 1],
           [1, 0, 0, 0, 1],
           [1, 1, 1, 1, 1]])
    >>> image = np.zeros((8, 8), dtype=int)
    >>> _set_border_values(image, 1, border_width=((1, 1), (2, 3)))
    >>> image
    array([[1, 1, 1, 1, 1, 1, 1, 1],
           [1, 1, 0, 0, 0, 1, 1, 1],
           [1, 1, 0, 0, 0, 1, 1, 1],
           [1, 1, 0, 0, 0, 1, 1, 1],
           [1, 1, 0, 0, 0, 1, 1, 1],
           [1, 1, 0, 0, 0, 1, 1, 1],
           [1, 1, 0, 0, 0, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1]])
    """
    if np.isscalar(border_width):
        border_width = ((border_width, border_width),) * image.ndim
    elif len(border_width) != image.ndim:
        raise ValueError('length of `border_width` must match image.ndim')
    for axis, npad in enumerate(border_width):
        if len(npad) != 2:
            raise ValueError('each sequence in `border_width` must have '
                             'length 2')
        w_start, w_end = npad
        if w_start == w_end == 0:
            continue
        elif w_start == w_end == 1:
            # Index first and last element in the current dimension
            sl = (slice(None),) * axis + ((0, -1),) + (...,)
            image[sl] = value
            continue
        if w_start > 0:
            # set first w_start entries along axis to value
            sl = (slice(None),) * axis + (slice(0, w_start),) + (...,)
            image[sl] = value
        if w_end > 0:
            # set last w_end entries along axis to value
            sl = (slice(None),) * axis + (slice(-w_end, None),) + (...,)
            image[sl] = value
            
#from skimage._utils import _offsets_to_raveled_neighbors
def crop(ar, crop_width, copy=False, order='K'):
    """Crop array `ar` by `crop_width` along each dimension.
    Parameters
    ----------
    ar : array-like of rank N
        Input array.
    crop_width : {sequence, int}
        Number of values to remove from the edges of each axis.
        ``((before_1, after_1),`` ... ``(before_N, after_N))`` specifies
        unique crop widths at the start and end of each axis.
        ``((before, after),) or (before, after)`` specifies
        a fixed start and end crop for every axis.
        ``(n,)`` or ``n`` for integer ``n`` is a shortcut for
        before = after = ``n`` for all axes.
    copy : bool, optional
        If `True`, ensure the returned array is a contiguous copy. Normally,
        a crop operation will return a discontiguous view of the underlying
        input array.
    order : {'C', 'F', 'A', 'K'}, optional
        If ``copy==True``, control the memory layout of the copy. See
        ``np.copy``.
    Returns
    -------
    cropped : array
        The cropped array. If ``copy=False`` (default), this is a sliced
        view of the input array.
    """
    ar = np.array(ar, copy=False)

    if isinstance(crop_width, numbers.Integral):
        crops = [[crop_width, crop_width]] * ar.ndim
    elif isinstance(crop_width[0], numbers.Integral):
        if len(crop_width) == 1:
            crops = [[crop_width[0], crop_width[0]]] * ar.ndim
        elif len(crop_width) == 2:
            crops = [crop_width] * ar.ndim
        else:
            raise ValueError(
                f'crop_width has an invalid length: {len(crop_width)}\n'
                f'crop_width should be a sequence of N pairs, '
                f'a single pair, or a single integer'
            )
    elif len(crop_width) == 1:
        crops = [crop_width[0]] * ar.ndim
    elif len(crop_width) == ar.ndim:
        crops = crop_width
    else:
        raise ValueError(
            f'crop_width has an invalid length: {len(crop_width)}\n'
            f'crop_width should be a sequence of N pairs, '
            f'a single pair, or a single integer'
        )

    slices = tuple(slice(a, ar.shape[i] - b)
                   for i, (a, b) in enumerate(crops))
    if copy:
        cropped = np.array(ar[slices], order=order, copy=True)
    else:
        cropped = ar[slices]
    return cropped

def _offsets_to_raveled_neighbors(image_shape, footprint, center, order='C'):
    """Compute offsets to a samples neighbors if the image would be raveled.
    Parameters
    ----------
    image_shape : tuple
        The shape of the image for which the offsets are computed.
    footprint : ndarray
        The footprint (structuring element) determining the neighborhood
        expressed as an n-D array of 1's and 0's.
    center : tuple
        Tuple of indices to the center of `footprint`.
    order : {"C", "F"}, optional
        Whether the image described by `image_shape` is in row-major (C-style)
        or column-major (Fortran-style) order.
    Returns
    -------
    raveled_offsets : ndarray
        Linear offsets to a samples neighbors in the raveled image, sorted by
        their distance from the center.
    Notes
    -----
    This function will return values even if `image_shape` contains a dimension
    length that is smaller than `footprint`.
    Examples
    --------
    >>> _offsets_to_raveled_neighbors((4, 5), np.ones((4, 3)), (1, 1))
    array([-5, -1,  1,  5, -6, -4,  4,  6, 10,  9, 11])
    >>> _offsets_to_raveled_neighbors((2, 3, 2), np.ones((3, 3, 3)), (1, 1, 1))
    array([ 2, -6,  1, -1,  6, -2,  3,  8, -3, -4,  7, -5, -7, -8,  5,  4, -9,
            9])
    """
    raveled_offsets = _raveled_offsets_and_distances(
            image_shape, footprint=footprint, center=center, order=order
            )[0]

    return raveled_offsets

def _raveled_offsets_and_distances(
        image_shape,
        *,
        footprint=None,
        connectivity=1,
        center=None,
        spacing=None,
        order='C',
        ):
    """Compute offsets to neighboring pixels in raveled coordinate space.
    This function also returns the corresponding distances from the center
    pixel given a spacing (assumed to be 1 along each axis by default).
    Parameters
    ----------
    image_shape : tuple of int
        The shape of the image for which the offsets are being computed.
    footprint : array of bool
        The footprint of the neighborhood, expressed as an n-dimensional array
        of 1s and 0s. If provided, the connectivity argument is ignored.
    connectivity : {1, ..., ndim}
        The square connectivity of the neighborhood: the number of orthogonal
        steps allowed to consider a pixel a neighbor. See
        `scipy.ndimage.generate_binary_structure`. Ignored if footprint is
        provided.
    center : tuple of int
        Tuple of indices to the center of the footprint. If not provided, it
        is assumed to be the center of the footprint, either provided or
        generated by the connectivity argument.
    spacing : tuple of float
        The spacing between pixels/voxels along each axis.
    order : 'C' or 'F'
        The ordering of the array, either C or Fortran ordering.
    Returns
    -------
    raveled_offsets : ndarray
        Linear offsets to a samples neighbors in the raveled image, sorted by
        their distance from the center.
    distances : ndarray
        The pixel distances correspoding to each offset.
    Notes
    -----
    This function will return values even if `image_shape` contains a dimension
    length that is smaller than `footprint`.
    Examples
    --------
    >>> off, d = _raveled_offsets_and_distances(
    ...         (4, 5), footprint=np.ones((4, 3)), center=(1, 1)
    ...         )
    >>> off
    array([-5, -1,  1,  5, -6, -4,  4,  6, 10,  9, 11])
    >>> d[0]
    1.0
    >>> d[-1]  # distance from (1, 1) to (3, 2)
    2.236...
    """
    ndim = len(image_shape)
    if footprint is None:
        footprint = ndi.generate_binary_structure(
                rank=ndim, connectivity=connectivity
                )
    if center is None:
        center = tuple(s // 2 for s in footprint.shape)

    if not footprint.ndim == ndim == len(center):
        raise ValueError(
            "number of dimensions in image shape, footprint and its"
            "center index does not match")

    offsets = np.stack([(idx - c)
                        for idx, c in zip(np.nonzero(footprint), center)],
                       axis=-1)

    if order == 'F':
        offsets = offsets[:, ::-1]
        image_shape = image_shape[::-1]
    elif order != 'C':
        raise ValueError("order must be 'C' or 'F'")

    # Scale offsets in each dimension and sum
    ravel_factors = image_shape[1:] + (1,)
    ravel_factors = np.cumprod(ravel_factors[::-1])[::-1]
    raveled_offsets = (offsets * ravel_factors).sum(axis=1)

    # Sort by distance
    if spacing is None:
        spacing = np.ones(ndim)
    weighted_offsets = offsets * spacing
    distances = np.sqrt(np.sum(weighted_offsets**2, axis=1))
    sorted_raveled_offsets = raveled_offsets[np.argsort(distances)]
    sorted_distances = np.sort(distances)

    # If any dimension in image_shape is smaller than footprint.shape
    # duplicates might occur, remove them
    if any(x < y for x, y in zip(image_shape, footprint.shape)):
        # np.unique reorders, which we don't want
        _, indices = np.unique(sorted_raveled_offsets, return_index=True)
        sorted_raveled_offsets = sorted_raveled_offsets[np.sort(indices)]
        sorted_distances = sorted_distances[np.sort(indices)]

    # Remove "offset to center"
    sorted_raveled_offsets = sorted_raveled_offsets[1:]
    sorted_distances = sorted_distances[1:]

    return sorted_raveled_offsets, sorted_distances


def flood(image, seed_point, *, footprint=None, connectivity=None,
          tolerance=None):
    
    """Mask corresponding to a flood fill.
    Starting at a specific `seed_point`, connected points equal or within
    `tolerance` of the seed value are found.
    Parameters
    ----------
    image : ndarray
        An n-dimensional array.
    seed_point : tuple or int
        The point in `image` used as the starting point for the flood fill.  If
        the image is 1D, this point may be given as an integer.
    footprint : ndarray, optional
        The footprint (structuring element) used to determine the neighborhood
        of each evaluated pixel. It must contain only 1's and 0's, have the
        same number of dimensions as `image`. If not given, all adjacent pixels
        are considered as part of the neighborhood (fully connected).
    connectivity : int, optional
        A number used to determine the neighborhood of each evaluated pixel.
        Adjacent pixels whose squared distance from the center is larger or
        equal to `connectivity` are considered neighbors. Ignored if
        `footprint` is not None.
    tolerance : float or int, optional
        If None (default), adjacent values must be strictly equal to the
        initial value of `image` at `seed_point`.  This is fastest.  If a value
        is given, a comparison will be done at every point and if within
        tolerance of the initial value will also be filled (inclusive).
    Returns
    -------
    mask : ndarray
        A Boolean array with the same shape as `image` is returned, with True
        values for areas connected to and equal (or within tolerance of) the
        seed point.  All other values are False.
    Notes
    -----
    The conceptual analogy of this operation is the 'paint bucket' tool in many
    raster graphics programs.  This function returns just the mask
    representing the fill.
    If indices are desired rather than masks for memory reasons, the user can
    simply run `numpy.nonzero` on the result, save the indices, and discard
    this mask.
    Examples
    --------
    >>> from skimage.morphology import flood
    >>> image = np.zeros((4, 7), dtype=int)
    >>> image[1:3, 1:3] = 1
    >>> image[3, 0] = 1
    >>> image[1:3, 4:6] = 2
    >>> image[3, 6] = 3
    >>> image
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 1, 1, 0, 2, 2, 0],
           [0, 1, 1, 0, 2, 2, 0],
           [1, 0, 0, 0, 0, 0, 3]])
    Fill connected ones with 5, with full connectivity (diagonals included):
    >>> mask = flood(image, (1, 1))
    >>> image_flooded = image.copy()
    >>> image_flooded[mask] = 5
    >>> image_flooded
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 5, 5, 0, 2, 2, 0],
           [0, 5, 5, 0, 2, 2, 0],
           [5, 0, 0, 0, 0, 0, 3]])
    Fill connected ones with 5, excluding diagonal points (connectivity 1):
    >>> mask = flood(image, (1, 1), connectivity=1)
    >>> image_flooded = image.copy()
    >>> image_flooded[mask] = 5
    >>> image_flooded
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 5, 5, 0, 2, 2, 0],
           [0, 5, 5, 0, 2, 2, 0],
           [1, 0, 0, 0, 0, 0, 3]])
    Fill with a tolerance:
    >>> mask = flood(image, (0, 0), tolerance=1)
    >>> image_flooded = image.copy()
    >>> image_flooded[mask] = 5
    >>> image_flooded
    array([[5, 5, 5, 5, 5, 5, 5],
           [5, 5, 5, 5, 2, 2, 5],
           [5, 5, 5, 5, 2, 2, 5],
           [5, 5, 5, 5, 5, 5, 3]])
    """
    # Correct start point in ravelled image - only copy if non-contiguous
    image = np.asarray(image)
    if image.flags.f_contiguous is True:
        order = 'F'
    elif image.flags.c_contiguous is True:
        order = 'C'
    else:
        image = np.ascontiguousarray(image)
        order = 'C'

    # Shortcut for rank zero
    if 0 in image.shape:
        return np.zeros(image.shape, dtype=bool)

    # Convenience for 1d input
    try:
        iter(seed_point)
    except TypeError:
        seed_point = (seed_point,)

    seed_value = image[seed_point]
    seed_point = tuple(np.asarray(seed_point) % image.shape)

    assert footprint is not None
    # footprint = _resolve_neighborhood(
    #     footprint, connectivity, image.ndim, enforce_adjacency=False)
    center = tuple(s // 2 for s in footprint.shape)
    # Compute padding width as the maximum offset to neighbors on each axis.
    # Generates a 2-tuple of (pad_start, pad_end) for each axis.
    pad_width = [(np.max(np.abs(idx - c)),) * 2
                 for idx, c in zip(np.nonzero(footprint), center)]

    # Must annotate borders
    working_image = np.pad(image, pad_width, mode='constant',
                           constant_values=image.min())
    # Stride-aware neighbors - works for both C- and Fortran-contiguity
    ravelled_seed_idx = np.ravel_multi_index(
        [i + pad_start
         for i, (pad_start, pad_end) in zip(seed_point, pad_width)],
        working_image.shape,
        order=order
    )
    neighbor_offsets = _offsets_to_raveled_neighbors(
        working_image.shape, footprint, center=center,
        order=order)

    # Use a set of flags; see _flood_fill_cy.pyx for meanings
    flags = np.zeros(working_image.shape, dtype=np.uint8, order=order)
    _set_border_values(flags, value=2, border_width=pad_width)

    try:
        if tolerance is not None:
            # Check if tolerance could create overflow problems
            try:
                max_value = np.finfo(working_image.dtype).max
                min_value = np.finfo(working_image.dtype).min
            except ValueError:
                max_value = np.iinfo(working_image.dtype).max
                min_value = np.iinfo(working_image.dtype).min

            high_tol = min(max_value, seed_value + tolerance)
            low_tol = max(min_value, seed_value - tolerance)

            _flood_fill_tolerance(working_image.ravel(order),
                                  flags.ravel(order),
                                  neighbor_offsets,
                                  ravelled_seed_idx,
                                  seed_value,
                                  low_tol,
                                  high_tol)
        else:
            _flood_fill_equal(working_image.ravel(order),
                              flags.ravel(order),
                              neighbor_offsets,
                              ravelled_seed_idx,
                              seed_value)
    except TypeError:
        if working_image.dtype == np.float16:
            # Provide the user with clearer error message
            raise TypeError("dtype of `image` is float16 which is not "
                            "supported, try upcasting to float32")
        else:
            raise

    # Output what the user requested; view does not create a new copy.
    return crop(flags, pad_width, copy=False).view(bool)

def detect_sprites(a, c,
                   connectivity=1):
    assert c is not None
    detections = []
    
    a = np.copy(a)

    x, y = np.meshgrid(np.arange(-3,3+1),np.arange(-3,3+1))
    radius = connectivity
    footprint = (x*x+y*y<=radius*radius)*1

    next_index = -2
    while True:
        nz = np.nonzero(a == c)
        try: x0, y0 = nz[0][0], nz[1][0]
        except: break

        flood_mask = flood(a, (x0, y0), footprint=footprint)

        nz = np.nonzero(flood_mask)
        x, y = nz[0].min(), nz[1].min()
        mask = flood_mask[x:nz[0].max()+1, y:nz[1].max()+1]

        yield {"x": x, "y": y, "c": c, "mask": mask}
        
        next_index-=1

        a[flood_mask] = next_index
        
    


def detect_diagonals(a):
    # diagonal lines where both of the coordinates increase together
    increasing = [a[-1]*1]
    for x in range(a.shape[0]-2, -1, -1):
        previous = increasing[-1]
        previous = np.pad(previous,((0,1)), mode='constant')[1:]
        new = a[x]*1*(1+previous)

        # ensure maximal diagonal
        # zeroing out the previous ones if we have new one
        increasing[-1] *= (np.pad(new,((1,0)), mode='constant')[:-1] == 0)
        
        
        increasing.append(new)
        
    increasing = np.stack(list(reversed(increasing)))

    # now we do decreasing
    decreasing = [a[-1]*1]
    for x in range(a.shape[0]-2, -1, -1):
        previous = decreasing[-1]
        previous = np.pad(previous,((1,0)), mode='constant')[:-1]
        new = a[x]*1*(1+previous)

        # ensure maximal diagonal
        # zeroing out the previous ones if we have new one
        decreasing[-1] *= (np.pad(new,((0,1)), mode='constant')[1:] == 0)
        
        
        decreasing.append(new)
        
    decreasing = np.stack(list(reversed(decreasing)))

    return increasing, decreasing


def detect_rectangles(a, _w=None, _h=None):
    
    
    skip = False
    nrows, ncols = a.shape
    area_max = (0, [])
    w = numpy.zeros(dtype=int, shape=a.shape)
    h = numpy.zeros(dtype=int, shape=a.shape)
    best_area = numpy.zeros(dtype=int, shape=a.shape)
    best_dimensions = numpy.zeros(dtype=int, shape=(*a.shape, 2))
    for x in range(nrows-1,-1,-1):
        for y in range(ncols-1,-1,-1):
            if a[x][y] == skip:
                continue
            if x == nrows-1:
                w[x][y] = 1
            else:
                w[x][y] = w[x+1][y]+1
            if y == ncols-1:
                h[x][y] = 1
            else:
                h[x][y] = h[x][y+1]+1

    for x in range(nrows):
        for y in range(ncols):
            if a[x][y] == skip:
                continue

            if _w is None and x > 0 and h[x-1,y] >= h[x,y]:
                continue

            if _h is None and y > 0 and w[x,y-1] >= w[x,y]:
                continue

            if True:
                this_height = h[x,y]
                for this_width in range(1, w[x][y]+1):
                    x2 = x + this_width - 1
                    this_height = min(this_height, h[x2,y])
                    if _w is not None and this_width != _w: continue
                    if _h is not None:
                        if this_height < _h: continue
                        adjusted_height = _h
                    else:
                        adjusted_height = this_height
                    area = this_width * adjusted_height
                    
                    if area > best_area[x][y]:
                        best_area[x][y] = area
                        best_dimensions[x,y,0] = this_width
                        best_dimensions[x,y,1] = adjusted_height

                    if area > area_max[0]:
                        area_max = (area, [(x,y, this_width, adjusted_height)])


    return best_area, best_dimensions

if __name__ == '__main__':
    
    
    a = np.zeros((6,6))
    a[:, 3] = 1
    a[2, :] = 1
    a = a==1
    best_area, best_dimensions = detect_rectangles(a, _w=1, _h=None)
    print(a.T)
    print(best_area.T)
    print(best_dimensions[:,:,0].T)
    print(best_dimensions[:,:,1].T)

    a = np.eye(6)>0
    a[2, 2]=0
    ##a[0,3]=1
    a[1, 4]=1
    a[2, 5]=1
    a[0,2] = 1
    print()

    print((a*1).T)


    print(detect_diagonals(a)[0].T)
    print(detect_diagonals(a)[1].T)
    

    # for t in best_area[1]:
    #     x, y, w, h = t
    #     print(f"x={x}, y={y}, w={w}, h={h}")
    #     print(a[x:x+w, y:y+h].T)

    x, y = np.meshgrid(np.arange(-3,3+1),np.arange(-3,3+1))
    radius = 1.5
    footprint = (x*x+y*y<=radius*radius)*1
    f = flood(a, )
