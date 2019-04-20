from __future__ import print_function

import re
from inspect import isfunction
from collections import defaultdict

from . import uff_pb2 as uff_pb
from .data import FieldType, create_data
from .exceptions import UffException


def _resolve_ref(field, referenced_data):
    field_type = field.WhichOneof("data_oneof")
    if field_type == FieldType.ref:
        if field.ref in referenced_data:
            return _resolve_ref(referenced_data[field.ref], referenced_data)
        else:
            raise UffException("Unknown reference: %s" % field.ref)
    else:
        return field


class _Constraint(object):

    def __init__(self, func, priority):
        self._deleted = False
        self.priority = priority
        self.func = func

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


class _DeletedConstraint(_Constraint):

    def __init__(self):
        super(_DeletedConstraint, self).__init__(None, 0)
        self._deleted = True


class _SharedDescriptorOpMemory(object):

    def __init__(self):
        self.fields = set()
        self.extra_fields = set()

    def mark_field(self, field_name, _is_extra_field=False):
        fields = self.fields if not _is_extra_field else self.extra_fields
        fields.add(field_name)


class DescriptorOp(object):

    def __init__(self):
        self._deleted = False
        self._constraints = {}
        self._priority = 0

    def extend_descriptor_op(self, descriptor_op):
        self._deleted = descriptor_op._deleted
        for name, constraint in descriptor_op._constraints.items():
            if name in self._constraints:
                # the priorities must stay the same for constraints that are linked
                self._constraints[name].func = constraint.func
                self._constraints[name]._deleted = constraint._deleted
            else:
                self._constraints[name] = constraint

    def delete_constraint(self, name):
        self._constraints[name] = _DeletedConstraint()
        return self

    def _delete_field(self, field_name, _is_extra_field):
        constraint_prefix = "field_" if not _is_extra_field else "extra_field_"
        return self.delete_constraint(constraint_prefix + field_name)

    def delete_field(self, field_name):
        return self._delete_field(field_name, _is_extra_field=False)

    def delete_extra_field(self, field_name):
        return self._delete_field(field_name, _is_extra_field=True)

    def constraint(self, name, func, error, priority=None):
        def _constraint(node, fields, extra_fields, shared_mem):
            res = func(node, fields, extra_fields, shared_mem)
            if not res:
                raise error

        if priority is None:
            self._priority += 1
            priority = self._priority

        self._constraints[name] = _Constraint(_constraint, priority)
        return self

    def _field(self, field_type, field_name, default_value=None, _is_extra_field=False):
        constraint_prefix = "field_" if not _is_extra_field else "extra_field_"

        def _check_field(n, f, e, s):
            if _is_extra_field:  # swap fields and extra_fields
                e, f = f, e

            s.mark_field(field_name)
            if field_name not in f:
                if isfunction(default_value):
                    f[field_name] = default_value(n, f, e, s)
                else:
                    f[field_name] = create_data(default_value, field_type)
            return f[field_name].WhichOneof("data_oneof") == field_type

        self.constraint(constraint_prefix + field_name, _check_field,
                        UffException("%s had bad type or is not present" % field_name))
        self._constraints[constraint_prefix + field_name].type = field_type
        return self

    def field(self, field_type, field_name, default_value=None):
        return self._field(field_type, field_name, default_value, _is_extra_field=False)

    def extra_field(self, field_type, field_name, default_value=None):
        return self._field(field_type, field_name, default_value, _is_extra_field=True)

    def _field_enum(self, field_name, enum, optional=False, _is_extra_field=False):
        constraint_prefix = "field_" if not _is_extra_field else "extra_field_"

        def _check_field(n, f, e, s):
            if _is_extra_field:
                e, f = f, e

            s.mark_field(field_name, _is_extra_field)
            if field_name not in f:
                return optional
            return f[field_name].WhichOneof("data_oneof") == FieldType.s and f[field_name].s in enum

        self.constraint(constraint_prefix + field_name, _check_field,
                        UffException("%s had bad type or is not present" % field_name))

        self._constraints[constraint_prefix + field_name].type = FieldType.s
        return self

    def field_enum(self, field_name, enum, optional=False):
        return self._field_enum(field_name, enum, optional, _is_extra_field=False)

    def extra_field_enum(self, field_name, enum, optional=False):
        return self._field_enum(field_name, enum, optional, _is_extra_field=True)

    def _ref_field(self, field_name, _is_extra_field):
        constraint_prefix = "ref_field_" if not _is_extra_field else "ref_extra_field_"

        def _check_field(n, f, e, s):
            f = n.fields if not _is_extra_field else n.extra_fields

            if field_name not in f:
                return True  # an optional field must not be checked in that constraint
            return f[field_name].WhichOneof("data_oneof") == FieldType.ref

        self.constraint(constraint_prefix + field_name, _check_field,
                        UffException("%s is not a referenced data" % field_name))
        return self

    def ref_field(self, field_name):
        return self._ref_field(field_name, _is_extra_field=False)

    def ref_extra_field(self, field_name):
        return self._ref_field(field_name, _is_extra_field=True)

    def fieldOrders(self, size=-1):

        def _inputs_orders_size(n, f, e, s):
            exp_size = len(n.inputs) if size < 0 else size
            return exp_size == len(f["inputs_orders"].dim_orders_list.val) and exp_size > 0

        def _default_outputs_orders(n, f, e, s):
            return create_data(
                               [f["inputs_orders"].dim_orders_list.val[0]],
                               FieldType.dim_orders_list)

        return (self.field(FieldType.dim_orders_list, "inputs_orders")
                .constraint("inputs_orders_size", _inputs_orders_size,
                            UffException("Invalid number of inputs_orders"))
                .field(FieldType.dim_orders_list, "outputs_orders", _default_outputs_orders)
                .ref_field("inputs_orders").ref_field("outputs_orders"))

    def inputs_size(self, size):
        return self.constraint("inputs_size",
                               lambda n, f, e, s: len(n.inputs) == size,
                               UffException("Invalid number of inputs, expected: %d" % size),
                               priority=0)

    def has_inputs(self):
        return self.constraint("has_inputs", lambda n, f, e, _: len(n.inputs) > 0,
                               UffException("No inputs found"), priority=0)

    def get_field_type(self, field_name):
        try:
            return self._constraints["field_" + field_name].type
        # except:
        #     raise UffException("The field {} doesn't exist".format(field_name))
        # catching the exception just to pass pylinter. I think the exception
        # message looks good.
        except Exception:
            raise UffException("The field {} doesn't exist".format(field_name))

    def _check_node(self, node, fields, extra_fields):
        shared_mem = _SharedDescriptorOpMemory()
        for constraint in sorted(self._constraints.values(), key=lambda c: c.priority):
            if constraint._deleted:
                continue
            err = constraint(node, fields, extra_fields, shared_mem)
            if err:
                raise UffException(err)

        for field_name in fields.keys():
            if field_name not in shared_mem.fields:
                raise UffException("field %s unknown" % field_name)

        for field_name in extra_fields.keys():
            if field_name not in shared_mem.extra_fields:
                raise UffException("extra_field %s unknown" % field_name)

        return True


class _DeletedDescriptorOp(DescriptorOp):

    def __init__(self):
        super(_DeletedDescriptorOp, self).__init__()
        self._deleted = True


class Descriptor(object):

    def __init__(self, name, version, optional, desc_ops):
        self._desc_ops = defaultdict(_DeletedDescriptorOp, desc_ops)
        self.name = name
        self.version = version
        self.optional = optional
        self.descriptors_extended = []
        self._regexes_operators = []

    def __delitem__(self, op, desc_op):
        self.add_descriptor_op(op, desc_op)

    def __setitem__(self, op, desc):
        self.add_operator(op, desc)

    def __getitem__(self, op):
        value = self._desc_ops[op]
        if value._deleted:
            raise KeyError(op)
        return self._desc_ops[op]

    def __contains__(self, op):
        if op in self._desc_ops:
            return not self._desc_ops[op]._deleted
        return False

    def to_uff(self, debug=False):
        if self.name is None:
            raise UffException("The core descriptor cannot be serialized")
        return uff_pb.Descriptor(id=self.name, version=self.version, optional=self.optional)

    def extend_descriptor(self, descriptor):
        for op, desc_op in descriptor._desc_ops.items():
            self._desc_ops[op].extend_descriptor_op(desc_op)

        self.descriptors_extended.append(descriptor)
        self.descriptors_extended.extend(descriptor.descriptors_extended)
        self._regexes_operators.extend(descriptor._regexes_operators)
        return self

    def delete_descriptor_op(self, op):
        self._desc_ops[op]._deleted = True
        return self

    def add_descriptor_op(self, op, desc_op):
        self._desc_ops[op] = desc_op
        return self

    def add_regex_operator(self, regex):
        self._regexes_operators.append(regex)
        return self

    def check_node(self, node, referenced_data):
        if node.operation not in self:
            if not any(re.match(regex_op, node.operation) for regex_op in self._regexes_operators):
                raise UffException("Unknown operation %s" % node.operation)
            return True

        fields = {k: _resolve_ref(v, referenced_data) for k, v in node.fields.items()}
        extra_fields = {k: _resolve_ref(v, referenced_data) for k, v in node.extra_fields.items()}

        return self._desc_ops[node.operation]._check_node(node, fields, extra_fields)
