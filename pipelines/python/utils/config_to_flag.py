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
"""Converts a Config dataclass to a set of Click flags."""

import inspect
from dataclasses import fields
from enum import Enum
from typing import Union, get_args, get_origin

import click


def config_to_flag(cls, defaults=None):
    options = []
    defaults = defaults or {}

    help_text = {} if not hasattr(cls, "help") else cls.help()

    for field in fields(cls):
        if field.name.startswith("_"):
            # Skip private config fields.
            continue
        normalized_name = field.name.lower().replace("_", "-")
        # If field type is a union on multiple types, set the argument type
        # as the first not-None type.
        field_type = field.type
        if get_origin(field_type) is Union:
            none_type = type(None)
            field_type = next(
                t for t in get_args(field.type) if t is not none_type
            )

        # For enum fields, convert to a choice that shows all possible values.
        if inspect.isclass(field_type) and issubclass(field_type, Enum):
            field_type = click.Choice(field_type)

        options.append(
            click.option(
                f"--{normalized_name}",
                show_default=True,
                type=field_type,
                default=defaults.get(field.name, field.default),
                help=help_text.get(field.name),
            )
        )

    def apply_flags(func):
        for option in reversed(options):
            func = option(func)
        return func

    return apply_flags
