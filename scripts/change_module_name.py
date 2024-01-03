#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Change the module name of a pre-existing object to a new name.

Created on 11/28/2023 at 1:52 PM
Author: dzhi
"""
import pickle
import importlib
from pathlib import Path

# Find model directory to save model fitting results
model_dir = 'Y:/data/Cerebellum/ProbabilisticParcellationModel'
if not Path(model_dir).exists():
    model_dir = '/srv/diedrichsen/data/Cerebellum/ProbabilisticParcellationModel'
if not Path(model_dir).exists():
    model_dir = '/Volumes/diedrichsen_data$/data/Cerebellum/ProbabilisticParcellationModel'
if not Path(model_dir).exists():
    raise (NameError('Could not find model_dir'))

def update_module_name(old_object, old_package_name, new_package_name):
    """ Change the module name of a pre-existing object to a new name.
        This function is useful and only works when the two modules are
        identical but having different names. For example, when changing
        an object <generativeMRF.arrngements.ArrangeIndependent> with
        <HierarchBayesParcel.arrngements.ArrangeIndependent>

    Args:
        old_object: an object with old module name to be changed
        old_package_name: the old module name
        new_package_name: the new module name

    Returns:
        new_object: an object with new module name
    """
    # Get the class name and module name
    class_name = old_object.__class__.__name__
    module_name = old_object.__class__.__module__

    # Change the old module name to the new one
    new_module_name = module_name.replace(old_package_name,
                                          new_package_name)

    # Creat new object dynamically
    new_module = importlib.import_module(new_module_name)
    new_class = getattr(new_module, class_name)
    new_object = new_class.__new__(new_class)

    # Copy attributes from the old object to the new object
    new_object.__dict__ = old_object.__dict__.copy()

    for attr, value in new_object.__dict__.items():
        if attr == 'arrange':
            setattr(new_object, attr,
                    update_module_name(value, old_package_name,
                                       new_package_name))
        elif attr == 'emissions':
            setattr(new_object, attr,
                    [update_module_name(item, old_package_name,
                                        new_package_name)
                     for item in value])

    return new_object

def make_new_pickles(file_name, new_module_name='HierarchBayesParcel',
                     out_file=None):
    """ Change the module name of a pre-existing object to a
        new name. The object will be saved back to the pickle
        file if save is True. This function is useful and only
        works when the two modules are identical but having
        different names.

    Args:
        file_name: the file name of the object to be changed
        new_module_name: the new module name of the object

    Returns:
        The updated object
    """
    # Load the pickled object
    with open(file_name, 'rb') as file:
        object_list = pickle.load(file)

    for i, old_object in enumerate(object_list):
        # Get the old module name
        old_module_name = old_object.__class__.__module__.split('.')[0]

        # Recursively update module names in the object's attributes
        new_object = update_module_name(old_object, old_module_name,
                                        new_module_name)
        object_list[i] = new_object

    # Write in the updated object back to the pickle file
    if out_file is None:
        out_file = file_name

    with open(out_file, 'wb') as file:
        pickle.dump(object_list, file)


if __name__ == '__main__':
    # Find the example pickle file of the models to be changed
    folder_path = Path(model_dir + f'/Models/Models_04')
    pickle_files = folder_path.glob('*.pickle')
    for fname in pickle_files:
        try:
            with open(str(fname), 'rb') as f:
                this_obj = pickle.load(f)

            if this_obj[0].__class__.__module__.split('.')[0] == 'generativeMRF':
                print('Converting -- ' + str(fname))
                make_new_pickles(str(fname), new_module_name='HierarchBayesParcel')
                print('Done')

        except:
            print('error in: ' + str(fname))
            continue

    # Test the models in new pickle file if successfully changed
    # with open(fname + '_new.pickle', 'rb') as file:
    #     object_list = pickle.load(file)
    #
    # full_model = object_list[0]
    # Prob = full_model.marginal_prob()
    # print(Prob)