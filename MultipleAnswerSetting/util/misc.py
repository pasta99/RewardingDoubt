from contextlib import ContextDecorator
from typing import Any

class disable_caching(ContextDecorator):  # noqa: N801
    def __init__(self, model):
        self.model = model
        self.prev_value: Any = "UNSET"  # config values may be T/F/None

    def __enter__(self):
        self.prev_value = self.model.config.use_cache
        self.model.config.use_cache = False

    def __exit__(self, *exc):
        if self.prev_value != "UNSET":
            self.model.config.use_cache = self.prev_value
        self.prev_value = "UNSET"