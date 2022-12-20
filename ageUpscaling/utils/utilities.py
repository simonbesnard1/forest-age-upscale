from math import ceil
import multiprocessing as mp
import dask
from collections.abc import Iterable
from threadpoolctl import threadpool_limits
from rasterio.enums import Resampling

def _iter_pack(func, itterable, *args, **kwargs):
    out = []
    for itter in itterable:
        _o = [func, itter]
        if len(args) > 0:
            _o.append(args)
        if len(kwargs) > 0:
            _o.append(kwargs)
        out.append(tuple(_o))
    return out


def _iter_unpack(IN):
    func = IN[0]
    itter = IN[1]
    args = None
    kwargs = None
    if len(IN) > 2:
        for _in in IN[2:]:
            if type(_in) is tuple:
                args = _in
            if type(_in) is dict:
                kwargs = _in
    try:
        if (args is None) and (kwargs is None):
            return func(itter)
        elif (args is not None) and (kwargs is None):
            return func(itter, *args)
        elif (args is None) and (kwargs is not None):
            return func(itter, **kwargs)
        elif (args is not None) and (kwargs is not None):
            return func(itter, *args, **kwargs)
        else:
            raise ValueError("Issue with function args and kwargs.")
    except Exception as e:
        raise RuntimeError(func.__name__ + ' faild with args: ' +
                           ' '.join([repr(i) for i in IN[1:]])).with_traceback(e.__traceback__)


def _async_run(IN, njobs=1):
    if (njobs > 1) and (mp.current_process().name == 'MainProcess'):
        with dask.config.set(scheduler='single-threaded'), threadpool_limits(limits=1, user_api='blas'):
            chunksize = ceil(len(IN) / njobs)
            pool = mp.Pool(njobs)
            out = pool.map_async(_iter_unpack, IN, chunksize=chunksize)
            pool.close()
            pool.join()
            out = out.get()
    else:
        out = map(_iter_unpack, IN)
    return(list(out))


def async_run(func, iterable, njobs, *args, **kwargs):
    """async_run(func, itterable, *args, **kwargs, njobs=1)

    Runs func asyncronouly across njobs processes. *args and **kwargs will be passed onto the function.

    Parameters
    ----------
    func : function
        The first argument of the function must be the value that changes, corresponding to an item in itterable
    itterable : list or list like
        The values which change for each job
    *args : args
        Arguments that will be passed onto the function, but will be the same for each call
    *kwargs : kwargs
        Keyword arguments that will be passed onto the function, but will be the same for each call
    """
    assert type(njobs) is int, "njobs must be an iteger"
    assert isinstance(iterable, Iterable), "`iterable` must be iterable, e.g. list, set, or tuple"
    to_proc = _iter_pack(func, iterable, *args, **kwargs)
    return _async_run(to_proc, njobs=njobs)

def interpolate_worlClim(source_ds, 
                         target_ds,
                         method:str = 'linear'):
    
    resampled = source_ds.interp(
                                latitude = target_ds.latitude, 
                                longitude = target_ds.longitude,
                                method=method)
    
    
    if not (resampled.latitude.data == target_ds.latitude.data).all():
        raise ValueError("Failed to interpolate in the latitude axis")
        
    if not (resampled.longitude.data == target_ds.longitude.data).all():
        raise ValueError("Failed to interpolate in the longitude axis")
    
    return resampled

