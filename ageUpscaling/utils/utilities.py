#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sbesnard
@File    :   cube_utils.py
@Time    :   Mon Sep 26 10:47:17 2022
@Author  :   Code adapted from Jake Nelson async_run functions
@Version :   1.0
@Contact :   besnard@gfz-potsdam.de
@License :   (C)Copyright 2022-2023, GFZ-Potsdam
@Desc    :   Functions to run asynchronous tasks
"""

from typing import Callable, Any, List, Tuple

from math import ceil

import multiprocessing as mp
import dask

from collections.abc import Iterable
from threadpoolctl import threadpool_limits

def _iter_pack(func: Callable, 
               itterable: Iterable, 
               *args: Any, 
               **kwargs: Any) -> List[Tuple[Callable, Any, Any, Any]]:
    """Pack function and arguments into tuples.
    
    Parameters
    ----------
    func : Callable
        The function to be called.
    itterable : Iterable
        An iterable containing the arguments to be passed to `func`.
    *args : Any
        Additional arguments to be passed to `func`.
    **kwargs : Any
        Additional keyword arguments to be passed to `func`.
        
    Returns
    -------
    out : List[Tuple[Callable, Any, Any, Any]]
        A list of tuples containing `func` and the arguments to be passed to it.
    """
    out = []
    for itter in itterable:
        _o = [func, itter]
        if len(args) > 0:
            _o.append(args)
        if len(kwargs) > 0:
            _o.append(kwargs)
        out.append(tuple(_o))
    return out

def _iter_unpack(IN:tuple):
    """
    Unpack the input tuple and execute the function with the specified arguments and keyword arguments.

    Parameters
    ----------
    IN : tuple
        A tuple containing the function as the first element, followed by the arguments and keyword arguments to be passed to the function.

    Returns
    -------
    Any
        The result of executing the function with the specified arguments and keyword arguments.

    Raises
    ------
    ValueError
        If the input tuple does not contain the correct number of elements.
    RuntimeError
        If an error occurs while executing the function with the specified arguments and keyword arguments.

    """
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

def _async_run(IN:tuple, 
               njobs:int=1):
    """
    Asynchronously runs a list of functions with their respective input arguments.
    
    Parameters
    ----------
    IN : list
        A list of tuples containing the function to be run and its input arguments.
    njobs : int, optional
        The number of jobs to run in parallel. Default is 1.
        
    Returns
    -------
    list
        A list containing the output of each function run.
    """
    if njobs > 1:
        with dask.config.set(scheduler='single-threaded'), threadpool_limits(limits=1, user_api='blas'):
            chunksize = ceil(len(IN) / njobs)
            pool = mp.Pool(njobs)
            out = pool.map_async(_iter_unpack, IN, chunksize=chunksize)
            pool.close()
            pool.join()
            out = out.get()
    else:
        out = map(_iter_unpack, IN)
    return list(out)

def async_run(func: Callable, 
              iterable: Iterable, 
              njobs:int, 
              *args:Any,
              **kwargs:Any) -> List:
    """Runs `func` asynchronously across `njobs` processes. `*args` and `**kwargs` will be passed onto `func`.

    Parameters
    ----------
    func : Callable
        The function to be run asynchronously. The first argument of `func` must be an item in `iterable`.
    iterable : Iterable
        The values which change for each job.
    njobs : int
        The number of processes to use for running `func` asynchronously.
    *args : args
        Additional arguments to pass to `func`. These will be the same for each call to `func`.
    *kwargs : kwargs
        Additional keyword arguments to pass to `func`. These will be the same for each call to `func`.

    Returns
    -------
    List
        A list of the results from each call to `func`.
    """
    assert type(njobs) is int, "njobs must be an iteger"
    assert isinstance(iterable, Iterable), "`iterable` must be iterable, e.g. list, set, or tuple"
    to_proc = _iter_pack(func, iterable, *args, **kwargs)
    return _async_run(to_proc, njobs=njobs)

