from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from ..data_contracts import MeshTexturePreparation
from .assets import MeshTextureBudgets, DEFAULT_BUDGETS, MeshTextureError


class MeshTextureCache:
    """Per-window LRU of CPU preparations; inactive layers may be retired."""

    def __init__(self, budgets: MeshTextureBudgets = DEFAULT_BUDGETS):
        self.budgets = budgets
        self._layers: OrderedDict[str, MeshTexturePreparation] = OrderedDict()
        self._active: set[str] = set()
        self._presentation_bytes: dict[str, int] = {}

    @staticmethod
    def _bytes(results):
        results = tuple(results)
        buffers = {
            id(image["pixels"]): image["pixels"].nbytes
            for result in results
            for image in result["images"].values()
        }
        buffers.update({
            id(result["face_uv"]): result["face_uv"].nbytes
            for result in results if "face_uv" in result
        })
        return sum(buffers.values())

    @property
    def nbytes(self):
        return self._cache_bytes(self._layers)

    def _cache_bytes(self, layers):
        return self._bytes(layers.values()) + sum(
            self._presentation_bytes.get(layer_id, 0) for layer_id in layers
        )

    def presentation_bytes(self, layer_id: str) -> int:
        return self._presentation_bytes.get(layer_id, 0)

    def set_presentation_bytes(self, layer_id: str, size: int) -> None:
        if layer_id not in self._layers:
            raise KeyError(layer_id)
        if type(size) is not int or size < 0:
            raise ValueError("Presentation size must be a nonnegative integer")
        if (
            self._bytes((self._layers[layer_id],)) + size
            > self.budgets.layer_cache_bytes
            or self.nbytes - self.presentation_bytes(layer_id) + size
            > self.budgets.global_cache_bytes
        ):
            raise MeshTextureError(
                "texture_budget", "Derived UVs exceed the CPU cache budget"
            )
        self._presentation_bytes[layer_id] = size

    def get(self, layer_id: str) -> MeshTexturePreparation | None:
        result = self._layers.get(layer_id)
        if result is not None:
            self._layers.move_to_end(layer_id)
        return result

    def set_active(self, layer_id: str, active: bool) -> None:
        if layer_id not in self._layers:
            raise KeyError(layer_id)
        if active:
            self._active.add(layer_id)
        else:
            self._active.discard(layer_id)

    def preparation_capacity(self) -> int:
        active = {
            key: value for key, value in self._layers.items() if key in self._active
        }
        return min(
            self.budgets.decoded_cpu_bytes,
            self.budgets.layer_cache_bytes,
            self.budgets.global_cache_bytes - self._cache_bytes(active),
        )

    @contextmanager
    def transaction(self) -> Iterator[None]:
        layers = self._layers.copy()
        active = self._active.copy()
        presentation_bytes = self._presentation_bytes.copy()
        try:
            yield
        except Exception:
            self._layers = layers
            self._active = active
            self._presentation_bytes = presentation_bytes
            raise

    def reserve_preparation(self, decoded_bytes: int) -> tuple[str, ...]:
        """Retire inactive buffers before allocating a new preparation."""
        candidate = self._layers.copy()
        evicted = []
        for layer_id in tuple(candidate):
            if (
                self._cache_bytes(candidate) + decoded_bytes
                <= self.budgets.global_cache_bytes
            ):
                break
            if layer_id not in self._active:
                del candidate[layer_id]
                evicted.append(layer_id)
        if (
            self._cache_bytes(candidate) + decoded_bytes
            > self.budgets.global_cache_bytes
        ):
            raise MeshTextureError(
                "global_cache_budget",
                "Existing and pending CPU images exceed the global cache budget",
            )
        self._layers = candidate
        for layer_id in evicted:
            self._presentation_bytes.pop(layer_id, None)
        return tuple(evicted)

    def store(
        self, layer_id: str, result: MeshTexturePreparation, *, active: bool
    ) -> tuple[str, ...]:
        if self._bytes((result,)) > self.budgets.layer_cache_bytes:
            raise MeshTextureError(
                "texture_budget", "Layer cache exceeds its byte limit"
            )
        candidate = self._layers.copy()
        candidate[layer_id] = result
        candidate.move_to_end(layer_id)
        evicted = []
        previous_bytes = self.presentation_bytes(layer_id)
        for old_id in tuple(candidate):
            if self._cache_bytes(candidate) - previous_bytes <= self.budgets.global_cache_bytes:
                break
            if old_id != layer_id and old_id not in self._active:
                del candidate[old_id]
                evicted.append(old_id)
        if self._cache_bytes(candidate) - previous_bytes > self.budgets.global_cache_bytes:
            raise MeshTextureError(
                "global_cache_budget", "Active texture images cannot be evicted"
            )
        active_layers = self._active.difference(evicted)
        if active:
            active_layers.add(layer_id)
        else:
            active_layers.discard(layer_id)
        self._layers = candidate
        self._presentation_bytes.pop(layer_id, None)
        self._active = active_layers
        for old_id in evicted:
            self._presentation_bytes.pop(old_id, None)
        return tuple(evicted)

    def remove(self, layer_id: str) -> None:
        self._layers.pop(layer_id, None)
        self._active.discard(layer_id)
        self._presentation_bytes.pop(layer_id, None)

    def clear(self) -> None:
        self._layers.clear()
        self._active.clear()
        self._presentation_bytes.clear()
