import pytest
from asr.common.params import Params, unflatten
import os
from asr.exceptions import ConfigurationError

params_content = """
float_item: 3.2
int_item: 0
env_var: $HOME/home
nested1:
    nested2:
        bool_item1: true
        bool_item2: "off"
"""

params_dict = {
    'float_item': 3.2,
    'int_item': 0,
    'env_var': os.environ.get('HOME') + '/home',
    'nested1': {
        'nested2': {
            'bool_item1': True,
            'bool_item2': "off"
        }
    }
}

params_flat_dict = {
    'float_item': 3.2,
    'int_item': 0,
    'env_var': os.environ.get('HOME') + '/home',
    'nested1.nested2.bool_item1': True,
    'nested1.nested2.bool_item2': "off"
}


def test_unflatten():
    flat_dict = {'a.b': [1, 2, 3], 'c': 1, 'd.e.f': {'g.h': 'string'}}
    unflatten_dict = {
        'a': {
            'b': [1, 2, 3]
        },
        'c': 1,
        'd': {
            'e': {
                'f': {
                    'g.h': 'string'
                }
            }
        }
    }

    assert unflatten(flat_dict) == unflatten_dict


@pytest.fixture(scope='session')
def param_file(tmpdir_factory):
    f = tmpdir_factory.mktemp('temp').join('params.yml')
    f.write(params_content)
    return f


def test_load(param_file):
    params = Params.load(param_file, params_overrides='env_var: nothing')
    assert params['env_var'] == 'nothing'

    params = Params.load(param_file,
                         params_overrides='nested1.nested2.bool_item1: False')

    assert params['nested1']['nested2']['bool_item1'] == False

    params = Params.load(param_file, ext_vars={'HOME': '/some/fold'})

    assert params['env_var'] == '/some/fold/home'


def test_save(param_file, tmpdir_factory):
    save_f = tmpdir_factory.mktemp('temp').join('saved.yml')
    params = Params.load(param_file)
    params.save(str(save_f))

    params_s = Params.load(str(save_f))

    assert params.params == params_s.params


def test_params(param_file):
    params = Params.load(param_file)

    assert len(params) == 4

    assert params.as_dict() == params_dict
    assert params.as_flat_dict() == params_flat_dict

    assert params.pop_bool('float_item')
    assert not params.pop_bool('int_item')

    nested_1_params = params['nested1']

    assert isinstance(nested_1_params, Params)
    assert nested_1_params.history == 'nested1.'

    assert nested_1_params['nested2']['bool_item1'] == 1
    nested_1_params['nested2']['bool_item1'] = 0
    assert nested_1_params['nested2']['bool_item1'] == 0

    assert nested_1_params['nested2']['bool_item2'] == 'off'
    assert not nested_1_params['nested2'].pop_bool('bool_item2')

    assert len(nested_1_params) == 1

    del nested_1_params['nested2']

    with pytest.raises(ConfigurationError):
        nested_1_params.pop('nested2')

    assert nested_1_params.assert_empty('lorem')

    with pytest.raises(ConfigurationError):
        params.assert_empty('lorem')

    for i, item in enumerate(params):
        pass

    assert i == 1
    assert item == 'nested1'
