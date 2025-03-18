from warnings import warn
import numpy as np
from scipy import linalg
from skimage._shared.utils import _supported_float_type, check_nD, deprecated
from skimage.feature.corner import hessian_matrix, hessian_matrix_eigvals, _symmetric_image
from skimage.util import img_as_float, invert

def get_eig(hess, **kwargs):
    """return the eigenvalue and eigenvectors of a Hessian (symmetric)

    The following is from numpy.linalg.eigh

    Return the eigenvalues and eigenvectors of a Hermitian or symmetric matrix.
    Returns two objects, a 1-D array containing the eigenvalues of `a`, and
    a 2-D square array or matrix (depending on the input type) of the
    corresponding eigenvectors (in columns).
    Parameters
    ----------
    a : array_like, shape (M, M)
        A complex Hermitian or real symmetric matrix.
    UPLO : {'L', 'U'}, optional
        Specifies whether the calculation is done with the lower triangular
        part of `a` ('L', default) or the upper triangular part ('U').
    Returns
    -------
    w : ndarray, shape (M,)
        The eigenvalues, not necessarily ordered.
    v : ndarray, or matrix object if `a` is, shape (M, M)
        The column ``v[:, i]`` is the normalized eigenvector corresponding
        to the eigenvalue ``w[i]``.
    Raises
    ------
    LinAlgError
        If the eigenvalue computation does not converge.
    """
    return np.linalg.eigh(hess, **kwargs)

def compute_hessian_eigen(image, sigma, sorting='none',
                                mode='constant', cval=0):
    """
    Compute Hessian eigenvalues of nD images.
    For 2D images, the computation uses a more efficient, skimage-based
    algorithm.
    Parameters
    ----------
    image : (N, ..., M) ndarray
        Array with input image data.
    sigma : float
        Smoothing factor of image for detection of structures at different
        (sigma) scales.
    sorting : {'val', 'abs', 'none'}, optional
        Sorting of eigenvalues by values ('val') or absolute values ('abs'),
        or without sorting ('none'). Default is 'none'.
    mode : {'constant', 'reflect', 'wrap', 'nearest', 'mirror'}, optional
        How to handle values outside the image borders.
    cval : float, optional
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.
    Returns
    -------
    eigenvalues : (D, N, ..., M) ndarray
        Array with (sorted) eigenvalues of Hessian eigenvalues for each pixel
        of the input image.
    """

    # Convert image to float
    float_dtype = _supported_float_type(image.dtype)
    # rescales integer images to [-1, 1]
    image = img_as_float(image)
    # make sure float16 gets promoted to float32
    image = image.astype(float_dtype, copy=False)

    # Make nD hessian
    hessian_elements = hessian_matrix(image, sigma=sigma, order='rc',
                                      mode=mode, cval=cval)

    # Correct for scale
    hessian_elements = [(sigma ** 2) * e for e in hessian_elements]

    # Compute Hessian eigenvalues and eigenvectors
    hessian_eigenvalues, hessian_eigenvectors = hessian_matrix_eigen(hessian_elements)

    if sorting == 'abs':
        # Sort eigenvalues by absolute values in ascending order
        sort_indices = abs(hessian_eigenvalues).argsort(0)
        # hessian_eigenvalues = _sortbyabs(hessian_eigenvalues, axis=0)

    elif sorting == 'val':
        # Sort eigenvalues by values in ascending order
        sort_indices = hessian_eigenvalues.argsort(0)
        # hessian_eigenvalues = np.sort(hessian_eigenvalues, axis=0)

    hessian_eigenvalues = np.take_along_axis(hessian_eigenvalues, sort_indices, 0)
    hessian_eigenvectors = np.take_along_axis(hessian_eigenvectors, sort_indices[None,:,:], 1)

    # Return Hessian eigenvalues
    return hessian_eigenvalues, hessian_eigenvectors

def hessian_matrix_eigen(S_elems):
    matrices = _symmetric_image(S_elems)
    # eigvalsh returns eigenvalues in increasing order. We want decreasing
    evals, evecs =get_eig(matrices)
    # leading_axes =
    return np.transpose(evals, (evals.ndim - 1,) + tuple(range(evals.ndim - 1))), np.transpose(evecs, tuple(range(evecs.ndim - 2, evecs.ndim)) + tuple(range(evecs.ndim - 2)))

def frangi(image, sigmas=range(1, 10, 2), scale_range=None,
           scale_step=None, alpha=0.5, beta=0.5, gamma=15,
           black_ridges=True, mode='reflect', cval=0):
    """
    Filter an image with the Frangi vesselness filter.
    This filter can be used to detect continuous ridges, e.g. vessels,
    wrinkles, rivers. It can be used to calculate the fraction of the
    whole image containing such objects.
    Defined only for 2-D and 3-D images. Calculates the eigenvectors of the
    Hessian to compute the similarity of an image region to vessels, according
    to the method described in [1]_.
    Parameters
    ----------
    image : (N, M[, P]) ndarray
        Array with input image data.
    sigmas : iterable of floats, optional
        Sigmas used as scales of filter, i.e.,
        np.arange(scale_range[0], scale_range[1], scale_step)
    scale_range : 2-tuple of floats, optional
        The range of sigmas used.
    scale_step : float, optional
        Step size between sigmas.
    alpha : float, optional
        Frangi correction constant that adjusts the filter's
        sensitivity to deviation from a plate-like structure.
    beta : float, optional
        Frangi correction constant that adjusts the filter's
        sensitivity to deviation from a blob-like structure.
    gamma : float, optional
        Frangi correction constant that adjusts the filter's
        sensitivity to areas of high variance/texture/structure.
    black_ridges : boolean, optional
        When True (the default), the filter detects black ridges; when
        False, it detects white ridges.
    mode : {'constant', 'reflect', 'wrap', 'nearest', 'mirror'}, optional
        How to handle values outside the image borders.
    cval : float, optional
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.
    Returns
    -------
    out : (N, M[, P]) ndarray
        Filtered image (maximum of pixels across all scales).
    Notes
    -----
    Written by Marc Schrijver, November 2001
    Re-Written by D. J. Kroon, University of Twente, May 2009, [2]_
    Adoption of 3D version from D. G. Ellis, Januar 20017, [3]_
    See also
    --------
    meijering
    sato
    hessian
    References
    ----------
    .. [1] Frangi, A. F., Niessen, W. J., Vincken, K. L., & Viergever, M. A.
        (1998,). Multiscale vessel enhancement filtering. In International
        Conference on Medical Image Computing and Computer-Assisted
        Intervention (pp. 130-137). Springer Berlin Heidelberg.
        :DOI:`10.1007/BFb0056195`
    .. [2] Kroon, D. J.: Hessian based Frangi vesselness filter.
    .. [3] Ellis, D. G.: https://github.com/ellisdg/frangi3d/tree/master/frangi
    """
    if scale_range is not None and scale_step is not None:
        warn('Use keyword parameter `sigmas` instead of `scale_range` and '
             '`scale_range` which will be removed in version 0.17.',
             stacklevel=2)
        sigmas = np.arange(scale_range[0], scale_range[1], scale_step)

    # Check image dimensions
    check_nD(image, [2, 3])

    # Check (sigma) scales
    sigmas = _check_sigmas(sigmas)

    # Rescale filter parameters
    alpha_sq = 2 * alpha ** 2
    beta_sq = 2 * beta ** 2
    gamma_sq = 2 * gamma ** 2

    # Get image dimensions
    ndim = image.ndim

    # Invert image to detect dark ridges on light background
    if black_ridges:
        image = invert(image)

    float_dtype = _supported_float_type(image.dtype)

    # Generate empty (n+1)D arrays for storing auxiliary images filtered
    # at different (sigma) scales
    filtered_max = np.zeros(image.shape, dtype=float_dtype)
    lambdas_array = np.zeros_like(filtered_max, dtype=float_dtype)
    orientation = np.zeros((ndim,image.shape[0],image.shape[1]), dtype=float_dtype)

    # Filtering for all (sigma) scales
    for i, sigma in enumerate(sigmas):

        # Calculate (abs sorted) eigenvalues
        (lambda1, *lambdas), evacs = compute_hessian_eigen(image, sigma,
                                                  sorting='abs',
                                                  mode=mode, cval=cval)

        # Compute sensitivity to deviation from a plate-like
        # structure see equations (11) and (15) in reference [1]_
        r_a = np.inf if ndim == 2 else _divide_nonzero(*lambdas) ** 2

        # Compute sensitivity to deviation from a blob-like structure,
        # see equations (10) and (15) in reference [1]_,
        # np.abs(lambda2) in 2D, np.sqrt(np.abs(lambda2 * lambda3)) in 3D
        filtered_raw = np.abs(np.multiply.reduce(lambdas)) ** (1/len(lambdas))
        r_b = _divide_nonzero(lambda1, filtered_raw) ** 2

        # Compute sensitivity to areas of high variance/texture/structure,
        # see equation (12)in reference [1]_
        r_g = sum([lambda1 ** 2] + [lambdai ** 2 for lambdai in lambdas])

        # Compute output image for given (sigma) scale and store results in
        # (n+1)D matrices, see equations (13) and (15) in reference [1]_
        vals = ((1 - np.exp(-r_a / alpha_sq))
                             * np.exp(-r_b / beta_sq)
                             * (1 - np.exp(-r_g / gamma_sq)))

        lambdas_array = np.max(lambdas, axis=0)
        # Remove background
        vals[lambdas_array > 0] = 0

        m = vals > filtered_max
        orientation[:, m] = evacs[:, 0, m]
        filtered_max[m] = vals[m]

    # # Remove background
    # filtered_array[lambdas_array > 0] = 0

    # Return for every pixel the maximum value over all (sigma) scales
    return filtered_max, orientation

# def frangi(image, sigmas=range(1, 10, 2), scale_range=None,
#            scale_step=None, alpha=0.5, beta=0.5, gamma=None,
#            black_ridges=True, mode='reflect', cval=0):
#     """
#     Filter an image with the Frangi vesselness filter.
#     This filter can be used to detect continuous ridges, e.g. vessels,
#     wrinkles, rivers. It can be used to calculate the fraction of the
#     whole image containing such objects.
#     Defined only for 2-D and 3-D images. Calculates the eigenvectors of the
#     Hessian to compute the similarity of an image region to vessels, according
#     to the method described in [1]_.
#     Parameters
#     ----------
#     image : (N, M[, P]) ndarray
#         Array with input image data.
#     sigmas : iterable of floats, optional
#         Sigmas used as scales of filter, i.e.,
#         np.arange(scale_range[0], scale_range[1], scale_step)
#     scale_range : 2-tuple of floats, optional
#         The range of sigmas used.
#     scale_step : float, optional
#         Step size between sigmas.
#     alpha : float, optional
#         Frangi correction constant that adjusts the filter's
#         sensitivity to deviation from a plate-like structure.
#     beta : float, optional
#         Frangi correction constant that adjusts the filter's
#         sensitivity to deviation from a blob-like structure.
#     gamma : float, optional
#         Frangi correction constant that adjusts the filter's
#         sensitivity to areas of high variance/texture/structure.
#         The default, None, uses half of the maximum Hessian norm.
#     black_ridges : boolean, optional
#         When True (the default), the filter detects black ridges; when
#         False, it detects white ridges.
#     mode : {'constant', 'reflect', 'wrap', 'nearest', 'mirror'}, optional
#         How to handle values outside the image borders.
#     cval : float, optional
#         Used in conjunction with mode 'constant', the value outside
#         the image boundaries.
#     Returns
#     -------
#     out : (N, M[, P]) ndarray
#         Filtered image (maximum of pixels across all scales).
#     Notes
#     -----
#     Earlier versions of this filter were implemented by Marc Schrijver,
#     (November 2001), D. J. Kroon, University of Twente (May 2009) [2]_, and
#     D. G. Ellis (January 2017) [3]_.
#     See also
#     --------
#     meijering
#     sato
#     hessian
#     References
#     ----------
#     .. [1] Frangi, A. F., Niessen, W. J., Vincken, K. L., & Viergever, M. A.
#         (1998,). Multiscale vessel enhancement filtering. In International
#         Conference on Medical Image Computing and Computer-Assisted
#         Intervention (pp. 130-137). Springer Berlin Heidelberg.
#         :DOI:`10.1007/BFb0056195`
#     .. [2] Kroon, D. J.: Hessian based Frangi vesselness filter.
#     .. [3] Ellis, D. G.: https://github.com/ellisdg/frangi3d/tree/master/frangi
#     """
#     if scale_range is not None and scale_step is not None:
#         warn('Use keyword parameter `sigmas` instead of `scale_range` and '
#              '`scale_range` which will be removed in version 0.17.',
#              stacklevel=2)
#         sigmas = np.arange(scale_range[0], scale_range[1], scale_step)
#
#     check_nD(image, [2, 3])  # Check image dimensions.
#     sigmas = _check_sigmas(sigmas) # Check (sigma) scales
#     # Rescale filter parameters
#     alpha_sq = 2 * alpha ** 2
#     beta_sq = 2 * beta ** 2
#     gamma_sq = 2 * gamma ** 2
#     image = image.astype(_supported_float_type(image.dtype), copy=False)
#     if not black_ridges:  # Normalize to black ridges.
#         image = -image
#
#     # Generate empty array for storing maximum value
#     # from different (sigma) scales
#     filtered_max = np.zeros_like(image)
#     orientation = np.zeros(shape=(image.ndim,image.shape[0],image.shape[1]))
#     for sigma in sigmas:  # Filter for all sigmas.
#         evals, evecs = hessian_matrix_eigen(hessian_matrix(
#             image, sigma, mode=mode, cval=cval, use_gaussian_derivatives=True))
#         # Sort eigenvalues by magnitude.
#         sort_indices = abs(evals).argsort(0)
#         eigvals = np.take_along_axis(evals, sort_indices, 0)
#         eigvcts = np.take_along_axis(evecs, sort_indices, 0)
#         lambda1 = eigvals[0]
#         if image.ndim == 2:
#             lambda2, = np.maximum(eigvals[1:], 1e-10)
#             r_a = np.inf  # implied by eq. (15).
#             r_b = abs(lambda1) / lambda2  # eq. (15).
#         else:  # ndim == 3
#             lambda2, lambda3 = np.maximum(eigvals[1:], 1e-10)
#             r_a = lambda2 / lambda3  # eq. (11).
#             r_b = abs(lambda1) / np.sqrt(lambda2 * lambda3)  # eq. (10).
#         s = np.sqrt((eigvals ** 2).sum(0))  # eq. (12).
#         if gamma is None:
#             gamma = s.max() / 2
#             if gamma == 0:
#                 gamma = 1  # If s == 0 everywhere, gamma doesn't matter.
#         # Filtered image, eq. (13) and (15).  Our implementation relies on the
#         # blobness exponential factor underflowing to zero whenever the second
#         # or third eigenvalues are negative (we clip them to 1e-10, to make r_b
#         # very large).
#         vals = 1.0 - np.exp(-r_a**2 / (2 * alpha**2))  # plate sensitivity
#         vals *= np.exp(-r_b**2 / (2 * beta**2))  # blobness
#         vals *= 1.0 - np.exp(-s**2 / (2 * gamma**2))  # structuredness
#         m = np.where(vals > filtered_max)
#         orientation[:, m] = eigvcts[:, m]
#         filtered_max[m] = vals[m]
#         # filtered_max = np.maximum(filtered_max, vals)
#     return filtered_max  # Return pixel-wise max over all sigmas.

def _check_sigmas(sigmas):
    """Check sigma values for ridges filters.
    Parameters
    ----------
    sigmas : iterable of floats
        Sigmas argument to be checked
    Returns
    -------
    sigmas : ndarray
        input iterable converted to ndarray
    Raises
    ------
    ValueError if any input value is negative
    """
    sigmas = np.asarray(sigmas).ravel()
    if np.any(sigmas < 0.0):
        raise ValueError('Sigma values should be equal to or greater '
                         'than zero.')
    return sigmas

def _divide_nonzero(array1, array2, cval=1e-10):
    """
    Divides two arrays.
    Denominator is set to small value where zero to avoid ZeroDivisionError and
    return finite float array.
    Parameters
    ----------
    array1 : (N, ..., M) ndarray
        Array 1 in the enumerator.
    array2 : (N, ..., M) ndarray
        Array 2 in the denominator.
    cval : float, optional
        Value used to replace zero entries in the denominator.
    Returns
    -------
    array : (N, ..., M) ndarray
        Quotient of the array division.
    """

    # Copy denominator
    denominator = np.copy(array2)

    # Set zero entries of denominator to small value
    denominator[denominator == 0] = cval

    # Return quotient
    return np.divide(array1, denominator)

def _sortbyabs(array, axis=0):
    """
    Sort array along a given axis by absolute values.
    Parameters
    ----------
    array : (N, ..., M) ndarray
        Array with input image data.
    axis : int
        Axis along which to sort.
    Returns
    -------
    array : (N, ..., M) ndarray
        Array sorted along a given axis by absolute values.
    Notes
    -----
    Modified from: http://stackoverflow.com/a/11253931/4067734
    """

    # Create auxiliary array for indexing
    index = list(np.ix_(*[np.arange(i) for i in array.shape]))

    # Get indices of abs sorted array
    index[axis] = np.abs(array).argsort(axis)

    # Return abs sorted array
    return array[tuple(index)]
