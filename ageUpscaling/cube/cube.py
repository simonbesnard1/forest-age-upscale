import atexit
import xarray as xr
import numpy as np
import os
import zarr
import shutil
from ageUpscaling.utils.utilities import async_run
import dask
synchronizer = zarr.ProcessSynchronizer('.zarrsync')

def cleanup():
    if os.path.isdir('.zarrsync') and (len(os.listdir('.zarrsync')) == 0):
        shutil.rmtree('.zarrsync')

atexit.register(cleanup)

DEFAULT_DIMS = ('cluster', 'sample')

DEFAULT_COORDS = {'cluster': np.arange(1, 145),
                  'sample':np.arange(0, 65803)}

DEFAULT_CHUNKS = {'cluster': -1,
                  'sample': -1}

def new_cube(cube_location, coords = None, chunks = None):
    """new_cube(cube_location, sites = None, coords='default')
    
    Create a new zarr site cube.

    Parameters
    ----------
    cube_location : str
        location of cube
    sites : str or list of str
        a site code (e.g. DE-Hai) or list of site codes
    coords : str or dict
        a dictionary of coordinates to use

    Notes
    -----
    Default coords are:
    
    DEFAULT_COORDS = {'cluster': np.arange(1, 145),
                      'sample':np.arange(0, 65803)}
    """    

    _ds = []
    encoding = {}
    for dim in coords.keys():
        _ds.append( xr.DataArray(coords={dim:coords[dim]}, dims=[dim]).chunk(chunks[dim]).rename(dim+'bnk') )

    _ds = xr.merge(_ds)

    for dim in coords.keys():
        del(_ds[dim+'bnk'])

    _ds.to_zarr(cube_location, encoding=encoding, consolidated=True)

    return coords


class Cube():
    """Cube(cube_location, coords='default', chunks='default', njobs=1)

    Handles creation and updating of regularized site_cube zarr files.

    Cubes are build with default coordinates unless otherwise stated. Default coords are:

    DEFAULT_COORDS = {'cluster': np.arange(1, 145),
                      'sample':np.arange(0, 65803)}

    Parameters
    ----------
    cube_location : str
        Path to site_cube .zarr array, which will be created if it does not exist.

    coords : dictionary of coordinates
        `coords` will be passed to xarray.Dataset(), and any not defined DEFAULT_COORDS will be added.

    chunks : dictionary defining chunks
        `chunks` will be passed to xarray.Dataset(), and any not defined DEFAULT_CHUNKS will be added:

            DEFAULT_CHUNKS = {'cluster': -1,
                              'sample': -1}

    njobs : int
        Number of cores to use in parallel when writing data.

    """

    def _init_cube(self):
        """
        Reloads the zarr file. If one does not yet exists, an empty one will be created.
        """
        if os.path.isdir(self.cube_location)==False:
            coords = new_cube(self.cube_location, coords = self.coords, chunks = self.chunks)

        self.cube = xr.open_zarr(self.cube_location)

    def __init__(self, cube_location, coords='default', chunks='default', njobs=1):
        self.cube_location = cube_location

        if coords == 'default':
            self.coords = {}
        else:
            self.coords = {k: v for k, v in coords.items() if len(v.shape) > 0}

        for k, v in DEFAULT_COORDS.items():
            if k in self.coords.keys():
                continue
            else:
                self.coords[k] = np.sort(np.asanyarray(v))

        if chunks == 'default':
            self.chunks = {}
        else:
            self.chunks = chunks
        for k, v in DEFAULT_CHUNKS.items():
            if k in self.chunks.keys():
                continue
            else:
                self.chunks[k] = v

        self.cube = None
        self.njobs = njobs
        self._init_cube()

    def _init_zarr_variable(self, IN):
        """
        Initializes a new zarr variable in the data cube.
        """
        name, dims, attrs, dtype = IN
        dims = [dim for dim in DEFAULT_DIMS if dim in dims]
        if name not in self.cube.variables:
            xr.DataArray(
                dask.array.full(shape=[self.cube.coords[dim].size for dim in dims],
                                chunks=[self.chunks[dim] for dim in dims], fill_value=np.nan),
                coords={dim: self.cube.coords[dim] for dim in dims},
                dims=dims,
                name=name,
                attrs=attrs
            ).chunk({dim: self.chunks[dim] for dim in dims}).to_dataset().to_zarr(self.cube_location, mode='a')

    def init_variable(self, dataset, njobs=None, parallel=False):
        """init_variable(dataset)

        Initializes all dataset variables in the Cube.

        Parameters
        ----------
        dataset : xr.Dataset, xr.DataArray, dictionary, or tuple
            Must be either xr.Dataset or xr.DataArray objects,
            or a dictionary with of the form {var_name: dims}
            where var_name is a string and dims is a list of
            dimension names as stings.

        njobs : int, default is None
            Number of cores to use in parallel when writing data, None will
            result in the default njobs as defined in during initialization.

        """
        if njobs is None:
            njobs = self.njobs
        self._init_cube()
        if type(dataset) is xr.DataArray:
            self._init_zarr_variable((dataset.name, dataset.dims, dataset.attrs, dataset.dtype))
        elif type(dataset) is tuple:
            self._init_zarr_variable(dataset)
        else:
            to_proc = []
            if type(dataset) is xr.Dataset:
                for _var in set(dataset.variables) - set(dataset.coords):
                    to_proc.append((_var, dataset[_var].dims, dataset[_var].attrs, dataset[_var].dtype))
            elif type(dataset) is dict:
                for k, v in dataset.items():
                    if type(v) is str:
                        to_proc.append((k, v, None, np.float64))
                    elif type(v) is dict:
                        to_proc.append((k, v['dims'], v['attrs'], np.float64))
                    elif len(v) == 2:
                        to_proc.append((k, v[0], v[1], np.float64))
                    else:
                        raise ValueError(
                            "key:value pair must be constructed as one of: var_name:(dims, attrs),\
                                var_name:{dims:dims, attrs:attrs}, or var_name:dim")
            else:
                raise RuntimeError("dataset must be xr.Dataset, xr.DataArray, dictionary, or tuple")
            if parallel:
                # issue with cascading multiprocessing, maybe fix in the future.
                out = async_run(self._init_zarr_variable, to_proc, njobs)
            else:
                out = list(map(self._init_zarr_variable, to_proc))

        self.cube = xr.open_zarr(self.cube_location)
        
    def _update_cube_DataArray(self, da):
        """
        Updates a single DataArray in the zarr cube. Data must be pre-sorted.
        Inputs to the `update_cube` function ultimately are passed here.
        """
        
        try:
            _zarr = zarr.open_group(self.cube_location, synchronizer = synchronizer)[da.name]
        except ValueError as e:
            raise FileExistsError("cube_location already exists but is not a zarr group. Delete existing directory or choose a different cube_location: "+self.cube_location) from e
        
        idxs = tuple([np.where( np.isin(self.cube[dim].values, da[dim].values ) )[0] for dim in da.dims])

        if len(_zarr.shape) != len(da.shape):
            raise ValueError("Inconsistent dimensions. Array `{0}` to be saved has dimensions of {1}, but target dataset expected {2}.".format(da.name, da.dims, self.cube[da.name].dims))
        try:
            _zarr.set_orthogonal_selection(idxs, da.data)
        except Exception as e:
            raise RuntimeError("Failed to write variable to site_cube: "+str(da)) from e
        
    def _update(self, da, njobs=None):
        """
        Handles both Datasets and DataArrays.
        """
        if njobs is None:
            njobs = self.njobs
        update_function = self._update_cube_DataArray
        if type(da) is xr.Dataset:
            to_proc = [da[_var] for _var in (set(da.variables) - set(da.coords))]
            out = async_run(update_function, to_proc, njobs)
        elif type(da) is xr.DataArray:
            update_function(da)
        else:
            raise RuntimeError("Input must be xr.Dataset or xr.DataArray objects")


    def update_cube(self, da, njobs=None, initialize=True, is_sorted=False):
        """update_cube(da, njobs=None, initialize=True, is_sorted=True)

        update the site cube with the provided Dataset or DataArray.

        Parameters
        ----------
        da : Dataset or DataArray
            should contain the data to be updated to the cube
        njobs : int
            number of CPUs to use in parallel when updating,
            each variable will be updated in parallel
        initialize : bool
            set false to skip variable initialization,
            faster if variables are pre-initialized
        is_sorted : bool
            set false to skip dimension sorting,
            faster if arrays are already sorted by dimension

        Returns
        -------
        z : float
            describe what z is

        """
        if njobs is None:
            njobs = self.njobs
        if is_sorted == False:  # noqa: E712
            for dim in da.dims:
                da = da.sortby(dim)
        da = da.transpose(*[dim for dim in DEFAULT_DIMS if dim in da.dims])
        if initialize:
            self.init_variable(da, njobs=njobs)
        self._update(da, njobs=njobs)

