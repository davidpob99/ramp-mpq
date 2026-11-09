#  Copyright 2026 David Población Criado
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import rootutils

PROJECT_ROOT = rootutils.setup_root(
    search_from=__file__,
    indicator=".project-root",
    project_root_env_var=True,  # set PROJECT_ROOT, interpolated by the configs
    dotenv=True,  # load credentials from .env
    pythonpath=True,  # make `src` importable
    cwd=True,  # resolve relative paths against the repository root
)

__all__ = ["PROJECT_ROOT"]