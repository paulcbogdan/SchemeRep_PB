import functools
import inspect
import os
import pickle
import zlib
from copy import copy
from datetime import datetime
from pathlib import Path
from pickle import UnpicklingError
from time import time

import numpy as np
from colorama import Fore

PICKLE_CACHE = {}

def getVariableName(variable, globalVariables):
    # from: https://stackoverflow.com/questions/18425225/getting-the-name-of-a-variable-as-a-string
    """ Get Variable Name as String by comparing its ID to globals() Variables' IDs
        args:
            variable(var): Variable to find name for (Obviously this variable has to exist)
        kwargs:
            globalVariables(dict): Copy of the globals() dict (Adding to Kwargs allows this function to work properly when imported from another .py)
    """
    for globalVariable in globalVariables:
        if id(variable) == id(globalVariables[globalVariable]): # If our Variable's ID matches this Global Variable's ID...
            return globalVariable # Return its name from the Globals() dict


def obj2str(val):
    kwargs_str = ''
    if isinstance(val, np.ndarray):
        return ''
    elif isinstance(val, dict):
        for key2 in sorted(val.keys()):
            kwargs_str += obj2str(val[key2])

    elif isinstance(val, list):
        for val_ in val:
            kwargs_str += obj2str(val_)

    else:
        kwargs_str += f'{val}_'
    if len(kwargs_str) > 20:
        # pre_kwargs_str = kwargs_str
        kwargs_str = str(zlib.adler32(kwargs_str.encode()))
        # assert len(kwargs_str) < 21, f'{len(kwargs_str)=}: {kwargs_str}'
        # print(f'Hash: {kwargs_str} | Pre: {pre_kwargs_str}')

    return kwargs_str


def f2str(callback, kwargs=None):
    if kwargs is None:
        kwargs = {}
    signature = inspect.signature(callback) # functools.partial impacts sig
    for k, v in signature.parameters.items():
        if v.default is v.empty: continue # exclude args, only want kwargs
        if k not in kwargs: kwargs[k] = v.default # TODO: could toggle off?
    if kwargs is not None:
        kwargs_str = ''
        for key in sorted(kwargs.keys()):
            val = kwargs[key]
            kwargs_str += obj2str(val)
        kwargs_str = kwargs_str[:-1]
    return kwargs_str


def get_default_fp(args, kwargs, callback, cache_dir, verbose=0):
    if verbose > 0: print(f'Making default filepath: {kwargs=}')
    if args is not None:
        args_str = '_'.join(args)
    else:
        args_str = ''
    kwargs_str = f2str(callback, kwargs)
    if isinstance(callback, functools.partial):
        name = callback.func.__name__
    else:
        name = callback.__name__
    func_dir = f'{cache_dir}/{name}'
    Path(func_dir).mkdir(parents=True, exist_ok=True)
    filepath = f'{func_dir}/{args_str}_{kwargs_str}.pkl'
    # Path(cache_dir).mkdir(parents=True, exist_ok=True)
    # filepath = f'{cache_dir}/{name}_{args_str}_{kwargs_str}.pkl'
    if verbose > 0: print(f'Default pickle_wrap filepath: {filepath}')



    # bad_chars = [';', ':', '!', "*", " ", '{', '}', ',', '\'']
    # for char in bad_chars:
    #     filepath = filepath.replace(char, '')
    # print(f'Default pickle_wrap filepath: {filepath}')
    # quit()
    return filepath


def pickle_wrap(callback: object, filepath: object = None,
                args: object = None, kwargs: object = None,
                easy_override: object = False,
                verbose: object = 0, cache_dir: object = 'cache',
                dt_max: object = None, RAM_cache: object = False):
    '''
    :param filepath: File to which the callback output should be loaded (if already created)
                     or where the callback output should be saved
    :param callback: Function to be pickle_wrapped
    :param args: Arguments passed to function (often not necessary)
    :param kwargs: Kwargs passed to the function (often not necessary)
    :param easy_override: If true, then the callback will be performed and the previous .pkl
                          save will be overwritten (if it exists)
    :param verbose: If true, then some additional details will be printed (the name of the callback,
                    and the time needed to perform the function or load the .pkl)
    :return: Returns the output of the callback or the output saved in filepath
    '''
    kwargs = copy(kwargs) # don't want to modify outside
    if filepath is None:
        filepath = get_default_fp(args, kwargs, callback, cache_dir, verbose)
    if RAM_cache and filepath in PICKLE_CACHE:
        return PICKLE_CACHE[filepath]

    if verbose > 0:
        print(f'pickle_wrap: {filepath=}')
        print('\tFunction:', getVariableName(callback,
                                             globalVariables=globals().copy()))

    if os.path.isfile(filepath):
        made = os.path.getmtime(filepath)
        fn = os.path.basename(filepath)
        if dt_max is None:
            dt_max = datetime(2023, 11, 18, 14, 0, 0, 0)
        dt_min = datetime(2023, 11, 1, 14, 0, 0, 0)

        dt = datetime.fromtimestamp(made)
        if dt < dt_max and dt > dt_min:
            easy_override = True
            if verbose >= 0:
                print(f'File ({fn}) was made: {dt}')
                print('\tFile is old, overriding')

    if os.path.isfile(filepath) and not easy_override:
        try:
            start = time()
            with open(filepath, "rb") as file:
                pk = pickle.load(file)
                if verbose > 0: print(f'\tLoad time: {time()-start:.3f} s')
                if verbose == 0: print(f'Pickle loaded '
                                       f'({time() - start:.3f} s): '
                                       f'{filepath=}')
                if RAM_cache: PICKLE_CACHE[filepath] = pk
                return pk
        except (UnpicklingError, MemoryError, EOFError) as e:
            print(f'{Fore.RED}{e=}')
            print(f'\t{Fore.YELLOW}{callback=}')
            print(f'\t{Fore.YELLOW}{filepath=}{Fore.RESET}')


    if verbose > 0:
        print('Callback:',
              getVariableName(callback, globalVariables=globals().copy()))
    start = time()
    if args:
        output = callback(*args)
    elif kwargs:
        output = callback(**kwargs)
    else:
        output = callback()
    if verbose > 0:
        print(f'\tFunction time: {time()-start:.3f} s')
        # print('\tDumping to file name:', filepath)
    start = time()
    try:
        # print(f'{filepath=}')

        with open(filepath, "wb") as new_file:
            pickle.dump(output, new_file)
        if verbose == 0: print(f'Pickle wrapped ({time()-start:.3f} s): '
                              f'{filepath=}')
    except FileNotFoundError:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as new_file:
            pickle.dump(output, new_file)
    if verbose > 0: print(f'\tDump time: {time()-start:.3f} s')
    if RAM_cache: PICKLE_CACHE[filepath] = output

    return output
