#!/usr/bin/python
"""
convert_to_uff.py

Main script for doing uff conversions from
different frameworks.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import argparse
import uff
import os

def _replace_ext(path, ext):
    return os.path.splitext(path)[0] + ext

def process_cmdline_args():
    """
    Helper function for processing commandline arguments
    """
    parser = argparse.ArgumentParser(description="""Converts TensorFlow models to Unified Framework Format (UFF).""")

    parser.add_argument(
        "input_file",
        help="""path to input model (protobuf file of frozen GraphDef)""")

    parser.add_argument(
        '-l', '--list-nodes', action='store_true',
        help="""show list of nodes contained in input file""")

    parser.add_argument(
        '-t', '--text', action='store_true',
        help="""write a text version of the output in addition to the
        binary""")

    parser.add_argument(
        '--write_preprocessed', action='store_true',
        help="""write the preprocessed protobuf in addition to the
        binary""")

    parser.add_argument(
        '-q', '--quiet', action='store_true',
        help="""disable log messages""")

    parser.add_argument(
        '-d', '--debug', action='store_true',
        help="""Enables debug mode to provide helpful debugging output""")

    parser.add_argument(
        "-o", "--output",
        help="""name of output uff file""")

    parser.add_argument(
        "-O", "--output-node", default=[], action='append',
        help="""name of output nodes of the model""")

    parser.add_argument(
        '-I', '--input-node', default=[], action='append',
        help="""name of a node to replace with an input to the model.
        Must be specified as: "name,new_name,dtype,dim1,dim2,..."
        """)

    parser.add_argument(
        "-p", "--preprocessor",
        help="""the preprocessing file to run before handling the graph. This file must define a `preprocess` function that accepts a GraphSurgeon DynamicGraph as it's input. All transformations should happen in place on the graph, as return values are discarded""")

    args, _ = parser.parse_known_args()
    args.output = _replace_ext((args.output if args.output else args.input_file), ".uff")
    return args, _

def main():
    args, _ = process_cmdline_args()
    if not args.quiet:
        print("Loading", args.input_file)
    uff.from_tensorflow_frozen_model(
        args.input_file,
        output_nodes=args.output_node,
        preprocessor=args.preprocessor,
        input_node=args.input_node,
        quiet=args.quiet,
        text=args.text,
        list_nodes=args.list_nodes,
        output_filename=args.output,
        write_preprocessed=args.write_preprocessed,
        debug_mode=args.debug
    )

if __name__ == '__main__':
    main()
