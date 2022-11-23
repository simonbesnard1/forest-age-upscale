import os
import json
import yaml
import inspect
from argparse import ArgumentParser
import getpass
from datetime import datetime
import importlib
from typing import Dict, Any
import time
import numpy as np
from xarray.core.coordinates import DataArrayCoordinates, DatasetCoordinates
import xarray as xr
from splcClassifier.core.argument_parsing import from_argparse_args, get_init_arguments
from splcClassifier.core.variables import VarHolder

_SKIP_TYPES = [
    xr.Dataset,
    xr.DataArray,
    np.ndarray,
    DataArrayCoordinates,
    DatasetCoordinates]


def obj_to_dict(obj):
    out = {}
    for k in obj.__dir__():
        if (k[:2] != '__') and (type(getattr(obj, k, '')).__name__ not in ['method', 'function']) and (k not in ['_abc_impl']):
            out[k] = getattr(obj, k, '')
    return out


def json_load_experiment_method(save_dir, file_name, class_name, module_name):
    module = importlib.import_module(module_name)
    _class = getattr(module, class_name)
    return _class.load(save_dir, file_name=file_name)


class ExperimentModule(object):
    CUSTOM_SKIP_TYPES = []

    @classmethod
    def from_argparse_args(cls, args, **kwargs):
        """Create an instance from CLI arguments.

        The respective arguments will be filtered, exceeding arguments will be ignored. Note that kwargs are given
        priority over args in case of duplicates.
        """
        return from_argparse_args(cls, args, **kwargs)

    @classmethod
    def add_cl_arguments(cls, parent_parser: ArgumentParser) -> ArgumentParser:
        """Load additional arguments on existing parser.

        Adds arguments from `self.custom_cl_arguments`.

        Parameters
        ----------
        parent_parser : ArgumentParser
            An argument parser.

        Returns
        -------
        Argument parser.
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        return cls.custom_cl_arguments(parser)

    @staticmethod
    def custom_cl_arguments(parser: ArgumentParser) -> ArgumentParser:
        """Add custom, module-specific CLI arguments.

        Default behaviour is that no CLI arguments are added, override this method to add yours. Using default values
        is highly recommended.

        Example:
            # Subclass `ExperimentModule` and add module-specific CLI arguments.
            class MyModule(ExperimentModule):
                def custom_cl_arguments(parser: ArgumentParser) -> ArgumentParser:
                    parser.add_argument(
                        '--my_argument',
                        type=int,
                        default=16,
                        help='my custom argument that does absolutely nothing (default: 16)')
                    return parser

        PArameters
        ----------
            parser : Argument Parser
                an existing argument parser, custom arguments specified here will be added.

        Returns:
            ArgumentParser, now loaded with custom CLI arguments specified here.
        """
        return parser

    @property
    def module_name(self) -> str:
        """Returns the module name"""
        return self.__module__

    @property
    def class_name(self) -> str:
        """Returns the class name."""
        return self.__class__.__name__

    @property
    def hparams(self) -> Dict[str, Any]:
        """"Get the hyper-parameters."""
        hparams = {}
        for arg in get_init_arguments(type(self)):
            hparams.update({arg: getattr(self, arg)})

        return hparams

    @staticmethod
    def create_and_get_path(*loc, exist_ok=True, is_file_path=False):
        if len(loc) > 0:
            path = os.path.join(*loc)
        else:
            path = ''

        if is_file_path:
            create_path = os.path.dirname(path)
        else:
            create_path = path

        if not os.path.exists(create_path):
            os.makedirs(create_path, exist_ok=exist_ok)
        return path

    @staticmethod
    def json_decode(dct):
        if '__VarHolder__' in dct:
            return VarHolder(dct['__VarHolder__'][0], **dct['__VarHolder__'][1])
        if '__Variable__' in dct:
            return Variable(
                dct['__Variable__']['name'],
                **{k: v for k, v in dct['__Variable__']['variant'].items() if v != '__SKIP_TYPES__'})
        if '__slice__' in dct:
            return slice(*dct['__slice__'])
        if '__ExperimentModule__' in dct:
            return json_load_experiment_method(*dct['__ExperimentModule__'])
        return dct

    def custom_json_decode(self, dct):
        return self.json_decode(dct)

    def json_serializer(self, obj):
        type_name = obj.__class__.__name__

        if type(obj) is VarHolder:
            return {'__VarHolder__': (obj.__name__, obj._input)}
        if type(obj) is Variable:
            return {'__Variable__': obj_to_dict(obj)}
        if isinstance(obj, ExperimentModule):
            obj.save(os.path.join(self._save_dir, obj.class_name))
            return({'__ExperimentModule__': (obj._save_dir, obj._file_name, obj.class_name, obj.module_name)})
        if type(obj) is slice:
            return {'__slice__': (obj.start, obj.stop, obj.step)}
        if type(obj) in _SKIP_TYPES + self.CUSTOM_SKIP_TYPES:
            return '__SKIP_TYPES__'
        if inspect.ismethod(obj):
            return '__SKIP_TYPES__'

        raise TypeError(f"Object of type '{type_name}' is not JSON serializable")

    def custom_json_serializer(self, obj):
        """add docstring
        """
        type_name = obj.__class__.__name__
        return self.json_serializer(obj)

    def custom_save(self, to_dump, save_dir=None):
        """method specific saving procedures
        """
        return to_dump

    @staticmethod
    def save_at(*loc, file_name=''):
        if len(loc) > 0:
            if not os.path.isdir(os.path.join(*loc)):
                os.makedirs(os.path.join(*loc))
        else:
            loc = ['']
        return os.path.join(*loc, file_name)

    def custom_load(self, load_dir, class_def=None, method_args=None):
        """method specific loading procedures
        """
        return self

    def save(self, save_dir, file_name=None):

        save_file = self.get_storage_file(save_dir, file_name)
        if os.path.exists(save_file) and (file_name is None):
            i = 1
            file_name = self.class_name + '{i:03d}.json'
            while os.path.exists(self.get_storage_file(save_dir, file_name.format(i=i))):
                i += 1
            save_file = self.get_storage_file(save_dir, file_name.format(i=i))
        self._save_dir, self._file_name = os.path.split(save_file)

        to_dump = self.custom_save(obj_to_dict(self), save_dir)

        with open(save_file, 'x') as f:
            json.dump(to_dump, f, default=self.custom_json_serializer, indent=2)

    @classmethod
    def get_storage_file(cls, dir_path, file_name=None):
        if not isinstance(cls, type):
            cls = type(cls)
        classname = cls.__name__
        if file_name is None:
            file_name = classname + '.json'

        return cls.create_and_get_path(dir_path, file_name, is_file_path=True)

    @classmethod
    def load(cls, load_dir, file_name=None):
        """Load object.

        Args:
            load_dir:
                Directory to load object from.
            file_name:
                Optional filename to load from. Default, this is classname + .json
        """

        load_file = cls.get_storage_file(load_dir, file_name)

        if load_file.split('.')[-1] in ["yml", "yaml"]:
            with open(load_file, 'r') as f:
                class_def = yaml.safe_load(f)
        else:
            with open(load_file, 'r') as f:
                class_def = json.load(f, object_hook=cls.json_decode)

        expected_args = inspect.getfullargspec(cls)
        method_args = {}
        for arg in expected_args.args:
            if arg in class_def.keys():
                method_args[arg] = class_def[arg]

        if expected_args.varargs is not None:
            class_obj = cls(*class_def.pop(expected_args.varargs), **method_args)
        else:
            class_obj = cls(**method_args)
        other_attrs = {k: v for k, v in class_def.items() if k not in method_args.keys()}

        for k, v in other_attrs.items():
            if v == '__SKIP_TYPES__':
                continue
            elif isinstance(getattr(type(class_obj), k, ''), property):
                continue
            else:
                setattr(class_obj, k, v)

        return class_obj.custom_load(load_dir, class_def=class_def, method_args=None)


def get_user_time(time_pattern='%Y%m%d'):
    user = getpass.getuser()
    time = datetime.now().strftime(time_pattern)
    return user, time


class TimeKeeper:
    
    def __init__(self, n_folds=1):
        self.start_time = time.time()
        self.lap_time   = time.time()
        self.n_folds    = n_folds
        self.counter    = 0
        self.lap_history = []
        
    def lap(self, message="{lap_time}", print_flag=True, skip_history=False):
        time_to_run = time.time() - self.lap_time
        lap_time    = "{0:.2f} {1}".format( time_to_run/60 if time_to_run>60 else time_to_run,
                                                        'min' if time_to_run>60 else 'sec')
        if print_flag:
            print(message.format(lap_time=lap_time))
            
        if not skip_history:
            self.lap_history.append(time_to_run)
        
        self.lap_time   = time.time()
        self.counter += 1
        return lap_time
    
    def total_time(self, message="{total_time}", print_flag=True):
        time_to_run = time.time() - self.start_time
        total_time  = "{0:.2f} {1}".format( time_to_run/60 if time_to_run>60 else time_to_run,
                                                        'min' if time_to_run>60 else 'sec')
        if print_flag:
            print(message.format(total_time=total_time))

        return total_time        
    
    def time_left(self, message="{total_time}, est. remaining: {time_left}", print_flag=True):
        total_time = self.total_time(print_flag=False)
        mean_time  = np.mean(self.lap_history)
        time_to_run = (mean_time*self.n_folds) - (mean_time*self.counter)
        time_left = "{0:.2f} {1}".format( time_to_run/60 if time_to_run>60 else time_to_run,
                                                        'min' if time_to_run>60 else 'sec')
        if print_flag:
            print(message.format(total_time=total_time, time_left=time_left))

        return time_left
