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

        See Also
        --------
        my_other_func : does something similar, but different

        Notes            y_data = self.method.get_y(self.provider, self.data_config)
            y_data = y_data.unstack().to_dataset('target')
            _coords = y_data.drop([k for k in y_data.coords if k not in y_data.dims]).coords
            y_attrs = {}
            for _y_var in y_data:
                y_attrs[_y_var] = self.provider.specify_variable(self.data_config.target._input[_y_var]).variant
                y_data[_y_var]  = y_data[_y_var].assign_attrs(**y_attrs[_y_var])
            _xval_cube = Cube(
                "/scratch/dMRVCH4_tmp/Cube_check",
                coords=_coords
            )
            _xval_cube.update_cube(
                y_data,
                njobs=self.n_jobs,
                initialize=True,
                is_sorted=False)
        -----
        This is an example of how to document a function. This also works for
        other objects.

        Examples
        --------
        >>> my_func(1, [1, 2], 'hello')  # noqa: F821
        2.5
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

