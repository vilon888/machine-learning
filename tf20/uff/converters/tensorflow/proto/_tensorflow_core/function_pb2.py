# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: _tensorflow_core/function.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from _tensorflow_core import attr_value_pb2 as __tensorflow__core_dot_attr__value__pb2
from _tensorflow_core import node_def_pb2 as __tensorflow__core_dot_node__def__pb2
from _tensorflow_core import op_def_pb2 as __tensorflow__core_dot_op__def__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='_tensorflow_core/function.proto',
  package='_tensorflow',
  syntax='proto3',
  serialized_pb=_b('\n\x1f_tensorflow_core/function.proto\x12\x0b_tensorflow\x1a!_tensorflow_core/attr_value.proto\x1a\x1f_tensorflow_core/node_def.proto\x1a\x1d_tensorflow_core/op_def.proto\"l\n\x12\x46unctionDefLibrary\x12*\n\x08\x66unction\x18\x01 \x03(\x0b\x32\x18._tensorflow.FunctionDef\x12*\n\x08gradient\x18\x02 \x03(\x0b\x32\x18._tensorflow.GradientDef\"\xaf\x02\n\x0b\x46unctionDef\x12%\n\tsignature\x18\x01 \x01(\x0b\x32\x12._tensorflow.OpDef\x12\x30\n\x04\x61ttr\x18\x05 \x03(\x0b\x32\"._tensorflow.FunctionDef.AttrEntry\x12&\n\x08node_def\x18\x03 \x03(\x0b\x32\x14._tensorflow.NodeDef\x12.\n\x03ret\x18\x04 \x03(\x0b\x32!._tensorflow.FunctionDef.RetEntry\x1a\x43\n\tAttrEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12%\n\x05value\x18\x02 \x01(\x0b\x32\x16._tensorflow.AttrValue:\x02\x38\x01\x1a*\n\x08RetEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\";\n\x0bGradientDef\x12\x15\n\rfunction_name\x18\x01 \x01(\t\x12\x15\n\rgradient_func\x18\x02 \x01(\tB/\n\x18org.tensorflow.frameworkB\x0e\x46unctionProtosP\x01\xf8\x01\x01\x62\x06proto3')
  ,
  dependencies=[__tensorflow__core_dot_attr__value__pb2.DESCRIPTOR,__tensorflow__core_dot_node__def__pb2.DESCRIPTOR,__tensorflow__core_dot_op__def__pb2.DESCRIPTOR,])
_sym_db.RegisterFileDescriptor(DESCRIPTOR)




_FUNCTIONDEFLIBRARY = _descriptor.Descriptor(
  name='FunctionDefLibrary',
  full_name='_tensorflow.FunctionDefLibrary',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='function', full_name='_tensorflow.FunctionDefLibrary.function', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='gradient', full_name='_tensorflow.FunctionDefLibrary.gradient', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=147,
  serialized_end=255,
)


_FUNCTIONDEF_ATTRENTRY = _descriptor.Descriptor(
  name='AttrEntry',
  full_name='_tensorflow.FunctionDef.AttrEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='_tensorflow.FunctionDef.AttrEntry.key', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='value', full_name='_tensorflow.FunctionDef.AttrEntry.value', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=_descriptor._ParseOptions(descriptor_pb2.MessageOptions(), _b('8\001')),
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=450,
  serialized_end=517,
)

_FUNCTIONDEF_RETENTRY = _descriptor.Descriptor(
  name='RetEntry',
  full_name='_tensorflow.FunctionDef.RetEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='_tensorflow.FunctionDef.RetEntry.key', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='value', full_name='_tensorflow.FunctionDef.RetEntry.value', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=_descriptor._ParseOptions(descriptor_pb2.MessageOptions(), _b('8\001')),
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=519,
  serialized_end=561,
)

_FUNCTIONDEF = _descriptor.Descriptor(
  name='FunctionDef',
  full_name='_tensorflow.FunctionDef',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='signature', full_name='_tensorflow.FunctionDef.signature', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='attr', full_name='_tensorflow.FunctionDef.attr', index=1,
      number=5, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='node_def', full_name='_tensorflow.FunctionDef.node_def', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='ret', full_name='_tensorflow.FunctionDef.ret', index=3,
      number=4, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[_FUNCTIONDEF_ATTRENTRY, _FUNCTIONDEF_RETENTRY, ],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=258,
  serialized_end=561,
)


_GRADIENTDEF = _descriptor.Descriptor(
  name='GradientDef',
  full_name='_tensorflow.GradientDef',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='function_name', full_name='_tensorflow.GradientDef.function_name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='gradient_func', full_name='_tensorflow.GradientDef.gradient_func', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=563,
  serialized_end=622,
)

_FUNCTIONDEFLIBRARY.fields_by_name['function'].message_type = _FUNCTIONDEF
_FUNCTIONDEFLIBRARY.fields_by_name['gradient'].message_type = _GRADIENTDEF
_FUNCTIONDEF_ATTRENTRY.fields_by_name['value'].message_type = __tensorflow__core_dot_attr__value__pb2._ATTRVALUE
_FUNCTIONDEF_ATTRENTRY.containing_type = _FUNCTIONDEF
_FUNCTIONDEF_RETENTRY.containing_type = _FUNCTIONDEF
_FUNCTIONDEF.fields_by_name['signature'].message_type = __tensorflow__core_dot_op__def__pb2._OPDEF
_FUNCTIONDEF.fields_by_name['attr'].message_type = _FUNCTIONDEF_ATTRENTRY
_FUNCTIONDEF.fields_by_name['node_def'].message_type = __tensorflow__core_dot_node__def__pb2._NODEDEF
_FUNCTIONDEF.fields_by_name['ret'].message_type = _FUNCTIONDEF_RETENTRY
DESCRIPTOR.message_types_by_name['FunctionDefLibrary'] = _FUNCTIONDEFLIBRARY
DESCRIPTOR.message_types_by_name['FunctionDef'] = _FUNCTIONDEF
DESCRIPTOR.message_types_by_name['GradientDef'] = _GRADIENTDEF

FunctionDefLibrary = _reflection.GeneratedProtocolMessageType('FunctionDefLibrary', (_message.Message,), dict(
  DESCRIPTOR = _FUNCTIONDEFLIBRARY,
  __module__ = '_tensorflow_core.function_pb2'
  # @@protoc_insertion_point(class_scope:_tensorflow.FunctionDefLibrary)
  ))
_sym_db.RegisterMessage(FunctionDefLibrary)

FunctionDef = _reflection.GeneratedProtocolMessageType('FunctionDef', (_message.Message,), dict(

  AttrEntry = _reflection.GeneratedProtocolMessageType('AttrEntry', (_message.Message,), dict(
    DESCRIPTOR = _FUNCTIONDEF_ATTRENTRY,
    __module__ = '_tensorflow_core.function_pb2'
    # @@protoc_insertion_point(class_scope:_tensorflow.FunctionDef.AttrEntry)
    ))
  ,

  RetEntry = _reflection.GeneratedProtocolMessageType('RetEntry', (_message.Message,), dict(
    DESCRIPTOR = _FUNCTIONDEF_RETENTRY,
    __module__ = '_tensorflow_core.function_pb2'
    # @@protoc_insertion_point(class_scope:_tensorflow.FunctionDef.RetEntry)
    ))
  ,
  DESCRIPTOR = _FUNCTIONDEF,
  __module__ = '_tensorflow_core.function_pb2'
  # @@protoc_insertion_point(class_scope:_tensorflow.FunctionDef)
  ))
_sym_db.RegisterMessage(FunctionDef)
_sym_db.RegisterMessage(FunctionDef.AttrEntry)
_sym_db.RegisterMessage(FunctionDef.RetEntry)

GradientDef = _reflection.GeneratedProtocolMessageType('GradientDef', (_message.Message,), dict(
  DESCRIPTOR = _GRADIENTDEF,
  __module__ = '_tensorflow_core.function_pb2'
  # @@protoc_insertion_point(class_scope:_tensorflow.GradientDef)
  ))
_sym_db.RegisterMessage(GradientDef)


DESCRIPTOR.has_options = True
DESCRIPTOR._options = _descriptor._ParseOptions(descriptor_pb2.FileOptions(), _b('\n\030org.tensorflow.frameworkB\016FunctionProtosP\001\370\001\001'))
_FUNCTIONDEF_ATTRENTRY.has_options = True
_FUNCTIONDEF_ATTRENTRY._options = _descriptor._ParseOptions(descriptor_pb2.MessageOptions(), _b('8\001'))
_FUNCTIONDEF_RETENTRY.has_options = True
_FUNCTIONDEF_RETENTRY._options = _descriptor._ParseOptions(descriptor_pb2.MessageOptions(), _b('8\001'))
# @@protoc_insertion_point(module_scope)
