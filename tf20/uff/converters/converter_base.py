"""
converter_base.py

Base class for all converters. Implement convert_layer
when inheriting from this class.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


class ConverterBase(object):
    """
    Base class for all converters. Creates a map of layers in the source
    framework, to the conversion functions for layers in the target framework.
    e.g. {'Conv': convert_conv(), 'FC': convert_fc()}
    """
    registry_ = {}

    @classmethod
    def register(cls, op_names=[]):
        """A decorator for registering gradient mappings."""

        def wrapper(func):
            """
            Creates a layer to function map
            """
            for op_name in op_names:
                cls.registry_[op_name] = func
            return func

        return wrapper

    @classmethod
    def convert_layer(cls, **kwargs):
        """Method for converting a layer"""
        pass
