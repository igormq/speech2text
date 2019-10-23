import copy
import logging
import os
import re
from collections.abc import MutableMapping

import yaml

from asr.exceptions import ConfigurationError
from asr.utils.io_utils import expandvars

logger = logging.getLogger(__name__)

path_matcher = re.compile(r'\$\{([^}^{]+)\}|\$([a-zA-Z0-9\_\-]+)|\%([^%^%]+)\%')


def recursively_expandvars(params, ext_vars={}):
    if isinstance(params, str):
        return expandvars(os.path.expanduser(params), environ=ext_vars)
    if isinstance(params, dict):
        return {k: recursively_expandvars(v, ext_vars=ext_vars) for k, v in params.items()}
    if isinstance(params, list):
        return [recursively_expandvars(p, ext_vars=ext_vars) for p in params]

    return params


def unflatten(flat_dict):
    """ Unflatten flatten dict.

    Given a "flattened" dict with compound keys, e.g.
        {"a.b": 0}
    unflatten it:
        {"a": {"b": 0}}
    """
    unflat_dict = {}

    for compound_key, value in flat_dict.items():
        curr_dict = unflat_dict
        parts = compound_key.split(".")
        for key in parts[:-1]:
            curr_value = curr_dict.get(key)
            if key not in curr_dict:
                curr_dict[key] = {}
                curr_dict = curr_dict[key]
            elif isinstance(curr_value, dict):
                curr_dict = curr_value
            else:
                raise ConfigurationError("flattened dictionary is invalid")
        if not isinstance(curr_dict, dict) or parts[-1] in curr_dict:
            raise ConfigurationError("flattened dictionary is invalid")
        else:
            curr_dict[parts[-1]] = value

    return unflat_dict


def merge_dicts(a, b, path=None):
    "merges b into a"
    if path is None: path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge_dicts(a[key], b[key], path + [str(key)])
            elif a[key] == b[key]:
                pass  # same leaf value
            else:
                # raise Exception('Conflict at %s' % '.'.join(path + [str(key)]))
                a[key] = b[key]
        else:
            a[key] = b[key]
    return a


class Params(MutableMapping):
    DEFAULT = object()

    def __init__(self, params, history="", ext_vars={}):
        self.params = params
        self.history = history
        self.ext_vars = ext_vars

    def get(self, key, default=DEFAULT):
        """
        Perform the functionality associated with dict.get(key) but also checks for returned
        dicts and returns a Params object in their place with an updated history.
        """
        if default is self.DEFAULT:
            try:
                value = self.params.get(key)
            except KeyError:
                raise ConfigurationError("key \"{}\" is required at location \"{}\"".format(key, self.history))
        else:
            value = self.params.get(key, default)

        if isinstance(value, str):
            value = recursively_expandvars(value, ext_vars=self.ext_vars)

        return self._check_is_dict(key, value)

    def pop(self, key, default=DEFAULT):
        if default is self.DEFAULT:
            try:
                value = self.params.pop(key)
            except KeyError:
                raise ConfigurationError(f"key '{key}' is required at location '{self.history}'")
        else:
            value = self.params.pop(key, default)

        if isinstance(value, str):
            value = recursively_expandvars(value, ext_vars=self.ext_vars)

        if not isinstance(value, dict):
            logger.info(self.history + key + " = " + str(value))

        return self._check_is_dict(key, value)

    def pop_int(self, key, default=DEFAULT):
        """Perform a pop and coerces to an int."""
        value = self.pop(key, default)

        if value is None:
            return None

        return int(value)

    def pop_float(self, key, default=DEFAULT):
        """Perform a pop and coerces to a float."""
        value = self.pop(key, default)

        if value is None:
            return None

        return float(value)

    def pop_bool(self, key, default=DEFAULT):
        """Perform a pop and coerces to a bool."""
        value = self.pop(key, default)

        if value is None:
            return None

        if isinstance(value, str):
            if value.lower() in ('true', 'yes', 'on', 'ok', '1'):
                return True

            if value.lower() in ('false', 'no', 'off', '0'):
                return False

        return bool(value)

    def pop_choice(self, key, choices):
        """
        Gets the value of ``key`` in the ``params`` dictionary, ensuring that the value is one of
        the given choices. Note that this `pops` the key from params, modifying the dictionary,
        consistent with how parameters are processed in this codebase.

        Params:
            key:
                Key to get the value from in the param dictionary
            choices:
                A list of valid options for values corresponding to ``key``.  For example,
                if you're specifying the type of encoder to use for some part of your model, the
                choices might be the list of encoder classes we know about and can instantiate.  If
                the value we find in the param dictionary is not in ``choices``, we raise a
                ``ConfigurationError``, because the user specified an invalid value in their
                parameter file.
        """
        value = self.pop(key, self.DEFAULT)
        if value not in choices:
            key_str = self.history + key
            message = '%s not in acceptable choices for %s: %s' % (value, key_str, str(choices))
            raise ConfigurationError(message)
        return value

    def as_dict(self, quiet=False):
        """Convert Params to dict.

        Params:
            quiet (bool, optional): Whether to log the parameters before returning them as a dict.
        """
        params_as_dict = self.params

        params_as_dict = recursively_expandvars(params_as_dict, ext_vars=self.ext_vars)

        if quiet:
            return params_as_dict

        logger.info("Converting Params object to dict; logging of default values will not occur "
                    "when dictionary parameters are used subsequently.")

        def log_recursively(parameters, history):
            for key, value in parameters.items():
                if isinstance(value, dict):
                    new_local_history = history + key + "."
                    log_recursively(value, new_local_history)
                else:
                    logger.info(history + key + " = " + str(value))

        logger.info("CURRENTLY DEFINED PARAMETERS: ")
        log_recursively(params_as_dict, self.history)
        return params_as_dict

    def as_flat_dict(self):
        """Convert Params to a flat dict.
        
        Returns the parameters of a flat dictionary from keys to values.
        Nested structure is collapsed with periods.
        """
        flat_params = {}

        def recurse(parameters, path):
            for key, value in parameters.items():
                newpath = path + [key]
                if isinstance(value, dict):
                    recurse(value, newpath)
                else:
                    flat_params['.'.join(newpath)] = value

        recurse(self.params, [])

        flat_params = recursively_expandvars(flat_params, ext_vars=self.ext_vars)
        return flat_params

    def duplicate(self):
        """Duplicate Params.
        
        Uses ``copy.deepcopy()`` to create a duplicate (but fully distinct)
        copy of these Params.
        """
        return Params(copy.deepcopy(self.params), history=self.history, ext_vars=self.ext_vars)

    def assert_empty(self, class_name):
        """Assert if Params is empty.
        Raises a ``ConfigurationError`` if ``self.params`` is not empty.  We take ``class_name`` as
        an argument so that the error message gives some idea of where an error happened, if there
        was one.  ``class_name`` should be the name of the `calling` class, the one that got extra
        parameters (if there are any).
        """
        if self.params:
            raise ConfigurationError("Extra parameters passed to {}: {}".format(class_name, self.params))

        return True

    def __getitem__(self, key):
        if key in self.params:
            return self._check_is_dict(key, self.params[key])
        else:
            raise KeyError

    def __setitem__(self, key, value):
        self.params[key] = value

    def __delitem__(self, key):
        del self.params[key]

    def __iter__(self):
        return iter(self.params)

    def __len__(self):
        return len(self.params)

    def _check_is_dict(self, new_history, value):
        if isinstance(value, dict):
            new_history = self.history + new_history + "."
            return Params(value, history=new_history, ext_vars=self.ext_vars)
        if isinstance(value, list):
            value = [self._check_is_dict(new_history + '.list', v) for v in value]
        if isinstance(value, str):
            value = recursively_expandvars(value, ext_vars=self.ext_vars)

        return value

    @staticmethod
    def load(params_file, params_overrides="{}", ext_vars=None):
        """Load a `Params` object from a configuration file.

        Params:
            params_file (str): The path to the configuration file to load.
            params_overrides  (str, optional): A dict of overrides that can be applied to final 
                object, e.g. '{"model.embedding_dim": 10}'.
            ext_vars (dict, optional): The config files allows specifying external variables
                for later substitution. Typically, it is substituted using environment
                variables; however, the user can also specify them here, in which case they
                take priority over environment variables.
                e.g. {"HOME_DIR": "/Users/someone/home"}
        """
        if ext_vars is None:
            ext_vars = {}

        ext_vars = {**os.environ, **ext_vars}

        with open(params_file, 'r', encoding='utf8') as stream:
            file_dict = yaml.full_load(stream)

        overrides_dict = unflatten(yaml.full_load(params_overrides))
        params_dict = merge_dicts(file_dict, overrides_dict)

        return Params(params_dict, ext_vars=ext_vars)

    def save(self, params_file):
        with open(params_file, "w") as handle:
            yaml.dump(self.params, handle)

    def __repr__(self):
        return repr(self.params)
