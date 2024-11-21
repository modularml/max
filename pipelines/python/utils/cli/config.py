# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Utilities for working with Config objects in Click."""

import functools
import inspect
import pathlib
from dataclasses import MISSING, fields
from enum import Enum
from pathlib import Path
from typing import Union, get_args, get_origin

import click
from max.driver import DeviceSpec
from max.pipelines import PipelineConfig, SupportedEncoding

from .device_options import DevicesOptionType


def config_to_flag(cls):
    options = []
    help_text = {} if not hasattr(cls, "help") else cls.help()
    for field in fields(cls):
        if field.name.startswith("_"):
            # Skip private config fields.
            continue
        normalized_name = field.name.lower().replace("_", "-")
        # If field type is a union on multiple types, set the argument type
        # as the first not-None type.
        field_type = field.type
        none_type = type(None)
        multiple = False
        if get_origin(field_type) is Union:
            field_type = next(
                t for t in get_args(field.type) if t is not none_type
            )
        elif get_origin(field_type) is list:
            field_type = next(
                t for t in get_args(field.type) if t is not none_type
            )
            multiple = True

        if inspect.isclass(field_type):
            # For enum fields, convert to a choice that shows all possible values.
            if issubclass(field_type, Enum):
                field_type = click.Choice(field_type)
            elif issubclass(field_type, pathlib.Path):
                field_type = click.Path(path_type=pathlib.Path)

        if field.default_factory != MISSING:
            default = field.default_factory()
        elif field.default != MISSING:
            default = field.default
        else:
            default = None
        options.append(
            click.option(
                f"--{normalized_name}",
                show_default=True,
                type=field_type,
                default=default,
                multiple=multiple,
                help=help_text.get(field.name),
            )
        )

    def apply_flags(func):
        for option in reversed(options):
            func = option(func)
        return func

    return apply_flags


def pipeline_config_options(func):
    @config_to_flag(PipelineConfig)
    @click.option(
        "--use-gpu",
        is_flag=False,
        type=DevicesOptionType(),
        show_default=False,
        default="",
        flag_value="0",
        help=(
            "Whether to run the model on the available GPU. An ID value can be"
            " provided optionally to indicate the device ID to target."
        ),
    )
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if kwargs["use_gpu"]:
            kwargs["device_spec"] = DeviceSpec.cuda(id=kwargs["use_gpu"][0])
            # If the user is passing in a specific, quantization_encoding don't overwrite it.
            # If it is empty, set it to default to bfloat16 on gpu.
            if kwargs["quantization_encoding"] is None:
                kwargs["quantization_encoding"] = SupportedEncoding.bfloat16
        else:
            kwargs["device_spec"] = DeviceSpec.cpu()

        del kwargs["use_gpu"]

        return func(*args, **kwargs)

    return wrapper
