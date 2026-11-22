from types import SimpleNamespace

import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import comfy.ldm.qwen_image21.model as qwen_image21  # noqa: E402
import comfy.memory_management  # noqa: E402
import comfy.model_management  # noqa: E402
import comfy.model_prefetch  # noqa: E402
import comfy.ops  # noqa: E402


class _Group:
    def __init__(self, linears):
        self.linears = linears
        self._v_weight = object()
        for m in linears:
            m._combined_linear = self


class _Module:
    def __init__(self, name, signature):
        self._v = name
        self._v_signature = signature
        self._combined_linear = None


def _module(name, signature="old"):
    return _Module(name, signature)


def _resident_module(name="resident"):
    s = _module(name, signature="same")
    s.weight = torch.empty((2, 2), device="meta")
    s.bias = None
    s._v_weight = torch.ones((2, 2))
    s._v_bias = None
    s._pin_state = {"fast_disk": False}
    return s


@pytest.fixture
def vbar(monkeypatch):
    state = SimpleNamespace(pinned=[], fault=None)
    monkeypatch.setattr(comfy.model_prefetch, "PREFETCH_QUEUES", [])
    monkeypatch.setattr(comfy.model_management, "PENDING_VBAR_CASTS", {})

    def fault(v):
        if state.fault is not None:
            signature = state.fault(v)
        else:
            signature = "same" if v.startswith("resident") else "new"
        if signature is not None:
            state.pinned.append(v)
        return signature

    def unpin(v):
        state.pinned.remove(v)

    model_vbar = comfy.ops.comfy_aimdo.model_vbar
    monkeypatch.setattr(model_vbar, "vbar_fault", fault)
    monkeypatch.setattr(model_vbar, "vbar_unpin", unpin)
    monkeypatch.setattr(model_vbar, "vbar_signature_compare", lambda a, b: a is not None and a == b)
    return state


@pytest.fixture
def failing_cast(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("VRAM grow failed")
    monkeypatch.setattr(comfy.ops, "materialize_meta_param", fail)


@pytest.fixture
def direct_cast(monkeypatch):
    monkeypatch.setattr(comfy.model_management, "is_device_cpu", lambda device: False)

    def cast(s, offloadable=True):
        return comfy.ops.cast_bias_weight(s, dtype=torch.float32, device=torch.device("cpu"), bias_dtype=torch.float32, offloadable=offloadable)

    return cast


def test_failed_cast_leaves_a_record_that_cleanup_releases(vbar, failing_cast):
    s = _module("weight")
    with pytest.raises(RuntimeError, match="VRAM grow failed"):
        comfy.ops.cast_modules_with_vbar([s], None, "cpu", None, False)
    assert vbar.pinned == ["weight"]
    assert s._prefetch == {"signature": "new", "resident": False}

    comfy.model_prefetch.cleanup_prefetched_modules(None, [s])
    assert vbar.pinned == []
    assert not hasattr(s, "_prefetch")


def test_cleanup_does_not_unpin_a_failed_fault(vbar, failing_cast):
    vbar.fault = lambda v: None
    s = _module("weight")
    with pytest.raises(RuntimeError, match="VRAM grow failed"):
        comfy.ops.cast_modules_with_vbar([s], None, "cpu", None, False)
    comfy.model_prefetch.cleanup_prefetched_modules(None, [s])
    assert vbar.pinned == []
    assert not hasattr(s, "_prefetch")


def test_batch_failure_cleanup_releases_every_faulted_module(vbar, failing_cast):
    resident = _module("resident", signature="same")
    missed = _module("missed")
    untouched = _module("untouched")
    with pytest.raises(RuntimeError, match="VRAM grow failed"):
        comfy.ops.cast_modules_with_vbar([resident, missed, untouched], None, "cpu", None, True, return_faulted=True)
    assert vbar.pinned == ["resident", "missed"]
    assert not hasattr(untouched, "_prefetch")

    comfy.model_prefetch.cleanup_prefetched_modules(None, [resident, missed, untouched])
    assert vbar.pinned == []


def test_group_failure_cleanup_releases_peers_faulted_up_front(vbar, failing_cast):
    members = [_module("a"), _module("b"), _module("c")]
    _Group(members)
    with pytest.raises(RuntimeError, match="VRAM grow failed"):
        comfy.ops.cast_modules_with_vbar(members, None, "cpu", None, True, return_faulted=True)
    assert vbar.pinned == ["a", "b", "c"]
    assert all(m._prefetch["resident"] is False for m in members)

    comfy.model_prefetch.cleanup_prefetched_modules(None, members)
    assert vbar.pinned == []
    assert all(not hasattr(m, "_prefetch") for m in members)


def test_fault_error_mid_group_cleanup_releases_earlier_peers(vbar):
    members = [_module("a"), _module("b"), _module("c")]
    _Group(members)

    def fault(v):
        if v == "c":
            raise RuntimeError("Fault failed: 2")
        return "new"
    vbar.fault = fault
    with pytest.raises(RuntimeError, match="Fault failed"):
        comfy.ops.cast_modules_with_vbar(members, None, "cpu", None, True, return_faulted=True)
    comfy.model_prefetch.cleanup_prefetched_modules(None, members)
    assert vbar.pinned == []


def test_prefetch_queue_cleanup_releases_a_partial_record(vbar, failing_cast):
    s = _module("weight")
    comfy.model_prefetch.PREFETCH_QUEUES.append([(None, (None, [s]))])
    with pytest.raises(RuntimeError, match="VRAM grow failed"):
        comfy.ops.cast_modules_with_vbar([s], None, "cpu", None, True, return_faulted=True)
    comfy.model_prefetch.cleanup_prefetch_queues()
    assert vbar.pinned == []
    assert not hasattr(s, "_prefetch")
    assert not comfy.model_prefetch.PREFETCH_QUEUES


def test_direct_cast_failure_releases_pin_and_record(vbar, failing_cast, monkeypatch):
    s = _module("weight")
    s.weight = torch.zeros(4)
    s.bias = None
    monkeypatch.setattr(comfy.model_management, "is_device_cpu", lambda device: False)
    with pytest.raises(RuntimeError, match="VRAM grow failed"):
        comfy.ops.cast_bias_weight(s, dtype=torch.float32, device=torch.device("meta"), bias_dtype=torch.float32, offloadable=True)
    assert vbar.pinned == []
    assert not hasattr(s, "_prefetch")
    assert not comfy.model_prefetch.PREFETCH_QUEUES


def test_direct_cast_resolve_failure_releases_pin_and_record(vbar, monkeypatch):
    s = _module("weight")
    s.weight = torch.zeros(4)
    s.bias = None

    def cast(modules, *args, **kwargs):
        for m in modules:
            m._prefetch = {"signature": comfy.ops.comfy_aimdo.model_vbar.vbar_fault(m._v), "resident": False}

    def resolve(*args, **kwargs):
        raise RuntimeError("Allocation on device failed")
    monkeypatch.setattr(comfy.model_management, "is_device_cpu", lambda device: False)
    monkeypatch.setattr(comfy.model_management, "sync_stream", lambda device, stream: None)
    monkeypatch.setattr(comfy.ops, "cast_modules_with_vbar", cast)
    monkeypatch.setattr(comfy.ops, "resolve_cast_module_with_vbar", resolve)
    with pytest.raises(RuntimeError, match="Allocation on device failed"):
        comfy.ops.cast_bias_weight(s, dtype=torch.float32, device=torch.device("meta"), bias_dtype=torch.float32, offloadable=True)
    assert vbar.pinned == []
    assert not hasattr(s, "_prefetch")
    assert not comfy.model_prefetch.PREFETCH_QUEUES


@pytest.mark.parametrize("offloadable", [False, True])
def test_global_cleanup_releases_direct_cast_after_layer_failure(vbar, direct_cast, offloadable):
    s = _resident_module()
    with pytest.raises(RuntimeError, match="mat1 and mat2 shapes"):
        weight, bias = direct_cast(s, offloadable=offloadable)[:2]
        torch.nn.functional.linear(torch.ones((1, 3)), weight, bias)

    assert vbar.pinned == ["resident"]
    assert not hasattr(s, "_prefetch")
    comfy.model_prefetch.cleanup_prefetch_queues()
    assert vbar.pinned == []
    assert not hasattr(s, "_prefetch")
    assert not comfy.model_prefetch.PREFETCH_QUEUES
    assert not comfy.model_management.PENDING_VBAR_CASTS


def test_global_cleanup_releases_first_cast_when_second_cast_fails(vbar, direct_cast, failing_cast):
    first = _resident_module()
    second = _module("missed")
    second.weight = torch.empty((2, 2), device="meta")
    second.bias = None

    direct_cast(first)
    with pytest.raises(RuntimeError, match="VRAM grow failed"):
        direct_cast(second)
    assert vbar.pinned == ["resident"]
    assert not hasattr(second, "_prefetch")

    comfy.model_prefetch.cleanup_prefetch_queues()
    assert vbar.pinned == []
    assert not comfy.model_prefetch.PREFETCH_QUEUES
    assert not comfy.model_management.PENDING_VBAR_CASTS


@pytest.mark.parametrize("cleanup", ["uncast", "global"])
@pytest.mark.parametrize("wait_fails", [False, True])
def test_direct_cast_cleanup_preserves_stream_wait_and_unpins_once(vbar, direct_cast, monkeypatch, cleanup, wait_fails):
    s = _resident_module()
    waited = []

    def wait(current):
        waited.append(current)
        if wait_fails:
            raise RuntimeError("stream wait failed")

    stream = SimpleNamespace(wait_stream=wait)
    current_stream = object()
    cast_modules = comfy.ops.cast_modules_with_vbar

    def cast(*args, **kwargs):
        cast_modules(*args, **kwargs)
        return stream

    monkeypatch.setattr(comfy.ops, "cast_modules_with_vbar", cast)
    monkeypatch.setattr(comfy.model_management, "sync_stream", lambda device, stream: None)
    monkeypatch.setattr(comfy.model_management, "current_stream", lambda device: current_stream)
    result = direct_cast(s)
    torch.testing.assert_close(torch.nn.functional.linear(torch.ones((1, 2)), *result[:2]), torch.full((1, 2), 2.0), rtol=0, atol=0)
    state = result[2]
    assert comfy.model_management.PENDING_VBAR_CASTS[id(state)][1] is state
    assert not hasattr(s, "_prefetch")
    assert not comfy.model_prefetch.PREFETCH_QUEUES

    def release():
        if cleanup == "uncast":
            comfy.ops.uncast_bias_weight(s, *result)
        else:
            comfy.model_prefetch.cleanup_prefetch_queues()

    if wait_fails:
        with pytest.raises(RuntimeError, match="stream wait failed"):
            release()
    else:
        release()
    assert waited == [current_stream]
    assert vbar.pinned == []
    assert not comfy.model_prefetch.PREFETCH_QUEUES
    assert not hasattr(s, "_prefetch")
    assert not comfy.model_management.PENDING_VBAR_CASTS

    comfy.model_prefetch.cleanup_prefetch_queues()
    comfy.ops.uncast_bias_weight(s, *result)
    assert vbar.pinned == []
    assert waited == [current_stream]


def test_direct_cast_context_cleans_up_when_layer_raises(vbar, monkeypatch):
    s = _resident_module()
    input = torch.ones((1, 3))
    monkeypatch.setattr(comfy.model_management, "is_device_cpu", lambda device: False)

    with pytest.raises(RuntimeError, match="mat1 and mat2 shapes"):
        with comfy.ops.CastBiasWeightContext(s, input, offloadable=True) as (weight, bias):
            assert not hasattr(s, "_prefetch")
            torch.nn.functional.linear(input, weight, bias)

    assert vbar.pinned == []
    assert not hasattr(s, "_prefetch")
    assert not comfy.model_prefetch.PREFETCH_QUEUES
    assert not comfy.model_management.PENDING_VBAR_CASTS


def test_direct_uncast_removes_only_its_cast(vbar, direct_cast):
    first = _resident_module("resident-first")
    second = _resident_module("resident-second")
    first_result = direct_cast(first)
    second_result = direct_cast(second)
    first_state = first_result[2]
    second_state = second_result[2]
    assert set(comfy.model_management.PENDING_VBAR_CASTS) == {id(first_state), id(second_state)}

    comfy.ops.uncast_bias_weight(first, *first_result)
    assert vbar.pinned == [second._v]
    assert id(first_state) not in comfy.model_management.PENDING_VBAR_CASTS
    assert comfy.model_management.PENDING_VBAR_CASTS[id(second_state)][1] is second_state
    comfy.model_prefetch.cleanup_prefetch_queues()
    assert vbar.pinned == []
    assert not comfy.model_prefetch.PREFETCH_QUEUES
    assert not comfy.model_management.PENDING_VBAR_CASTS


def test_prefetch_queue_keeps_ownership_of_prefetched_cast(vbar, direct_cast):
    s = _resident_module()
    comfy.ops.cast_modules_with_vbar([s], None, torch.device("cpu"), None, False)
    queue = [(None, (None, [s]))]
    comfy.model_prefetch.PREFETCH_QUEUES.append(queue)

    result = direct_cast(s)
    comfy.ops.uncast_bias_weight(s, *result)
    assert comfy.model_prefetch.PREFETCH_QUEUES == [queue]
    assert not comfy.model_management.PENDING_VBAR_CASTS
    assert result[2] == (None, None, None)
    assert vbar.pinned == ["resident"]
    assert hasattr(s, "_prefetch")

    comfy.model_prefetch.cleanup_prefetch_queues()
    assert vbar.pinned == []
    assert not hasattr(s, "_prefetch")
    assert not comfy.model_prefetch.PREFETCH_QUEUES


@pytest.mark.parametrize("signature", [None, "new"])
def test_direct_cast_miss_tracks_only_successful_fault(vbar, direct_cast, monkeypatch, signature):
    s = _resident_module()
    vbar.fault = lambda v: signature
    gathered = torch.empty(comfy.memory_management.vram_aligned_size(s.weight), dtype=torch.uint8)
    monkeypatch.setattr(comfy.ops.comfy_aimdo.torch, "aimdo_to_tensor", lambda v, device: gathered)
    monkeypatch.setattr(comfy.pinned_memory, "get_pin", lambda *args, **kwargs: None)
    monkeypatch.setattr(comfy.pinned_memory, "pin_memory", lambda *args, **kwargs: None)
    monkeypatch.setattr(comfy.model_management, "get_offload_stream", lambda device: None)

    result = direct_cast(s)
    torch.testing.assert_close(result[0], torch.zeros((2, 2)), rtol=0, atol=0)
    assert s._v_signature == signature
    assert vbar.pinned == ([s._v] if signature is not None else [])
    assert len(comfy.model_management.PENDING_VBAR_CASTS) == int(signature is not None)
    assert not comfy.model_prefetch.PREFETCH_QUEUES
    assert not hasattr(s, "_prefetch")
    comfy.ops.uncast_bias_weight(s, *result)
    comfy.model_prefetch.cleanup_prefetch_queues()
    assert vbar.pinned == []
    assert not hasattr(s, "_prefetch")
    assert not comfy.model_prefetch.PREFETCH_QUEUES
    assert not comfy.model_management.PENDING_VBAR_CASTS


def test_direct_cast_clears_prepared_patch_state_before_returning(vbar, direct_cast):
    s = _resident_module()
    cleared = []
    s.weight_lowvram_function = SimpleNamespace(clear_prepared=lambda: cleared.append("weight"))

    result = direct_cast(s)
    assert cleared == ["weight"]
    comfy.ops.uncast_bias_weight(s, *result)
    assert cleared == ["weight"]
    assert not hasattr(s, "_prefetch")
    assert vbar.pinned == []
    assert not comfy.model_prefetch.PREFETCH_QUEUES


@pytest.mark.parametrize("failure", ["graph", "queue"])
def test_global_cleanup_releases_direct_casts_when_other_cleanup_fails(vbar, direct_cast, monkeypatch, failure):
    s = _resident_module()
    real_cleanup = comfy.model_prefetch.cleanup_prefetched_modules

    def fail():
        raise RuntimeError("cleanup failed")

    if failure == "graph":
        monkeypatch.setattr(comfy.model_prefetch, "cleanup_malloc_graph", fail)
    else:
        failing = _module("failing")
        comfy.model_prefetch.PREFETCH_QUEUES.append([(None, (failing, []))])

        def cleanup(module, modules):
            if module is failing:
                fail()
            real_cleanup(module, modules)

        monkeypatch.setattr(comfy.model_prefetch, "cleanup_prefetched_modules", cleanup)
    direct_cast(s)

    with pytest.raises(RuntimeError, match="cleanup failed"):
        comfy.model_prefetch.cleanup_prefetch_queues()
    assert vbar.pinned == []
    assert not comfy.model_prefetch.PREFETCH_QUEUES
    assert not comfy.model_management.PENDING_VBAR_CASTS


def test_global_cleanup_continues_after_direct_unpin_failure(vbar, direct_cast, monkeypatch):
    first = _resident_module("resident-first")
    second = _resident_module("resident-second")
    direct_cast(first)
    direct_cast(second)
    unpin = comfy.ops.comfy_aimdo.model_vbar.vbar_unpin
    calls = []

    def failing_unpin(v):
        calls.append(v)
        if v == first._v:
            raise RuntimeError("unpin failed")
        unpin(v)

    monkeypatch.setattr(comfy.ops.comfy_aimdo.model_vbar, "vbar_unpin", failing_unpin)
    with pytest.raises(RuntimeError, match="unpin failed"):
        comfy.model_prefetch.cleanup_prefetch_queues()
    assert calls == [first._v, second._v]
    assert vbar.pinned == [first._v]
    assert not comfy.model_prefetch.PREFETCH_QUEUES
    assert not hasattr(first, "_prefetch")
    assert not hasattr(second, "_prefetch")
    assert not comfy.model_management.PENDING_VBAR_CASTS
    comfy.model_prefetch.cleanup_prefetch_queues()
    assert calls == [first._v, second._v]


def test_direct_cast_cleanup_does_not_call_prefetcher(vbar, direct_cast, failing_cast, monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("direct casts must not call prefetch cleanup")

    monkeypatch.setattr(comfy.model_prefetch, "cleanup_prefetched_modules", fail)
    monkeypatch.setattr(comfy.model_prefetch, "cleanup_prefetch_queues", fail)
    s = _resident_module()
    result = direct_cast(s)
    comfy.ops.uncast_bias_weight(s, *result)
    assert vbar.pinned == []

    direct_cast(s)
    comfy.model_management.cleanup_vbar_casts()
    assert vbar.pinned == []
    assert not comfy.model_management.PENDING_VBAR_CASTS

    missed = _module("missed")
    missed.weight = torch.empty((2, 2), device="meta")
    missed.bias = None
    with pytest.raises(RuntimeError, match="VRAM grow failed"):
        direct_cast(missed)
    assert vbar.pinned == []
    assert not hasattr(missed, "_prefetch")


def test_reset_cast_buffers_cleans_outstanding_direct_casts(vbar, direct_cast):
    s = _resident_module()
    direct_cast(s)

    comfy.model_management.reset_cast_buffers()

    assert vbar.pinned == []
    assert not comfy.model_management.PENDING_VBAR_CASTS


def test_prefix_cache_layer_cleans_slots_when_pinning_fails(monkeypatch):
    slot = SimpleNamespace(comfy_cast_weights=True)

    def pin_modules(modules, device):
        for m in modules:
            m._prefetch = {"signature": "new", "resident": False}
        raise RuntimeError("Pin registration failed")
    cleaned = []
    monkeypatch.setattr(comfy.model_prefetch, "pin_modules", pin_modules)
    monkeypatch.setattr(comfy.model_prefetch, "cleanup_prefetched_modules", lambda module, modules: cleaned.extend(modules))

    cache = qwen_image21.PrefixCache([[slot]])
    with pytest.raises(RuntimeError, match="Pin registration failed"):
        with cache.layer(0, "cpu", write=False):
            pass
    assert cleaned == [slot]


def test_resolve_failure_does_not_record_the_new_signature():
    s = _module("weight", signature="old")
    s.weight = torch.ones(4)
    s.bias = None
    geometry = comfy.memory_management.tensors_to_geometries([s.weight, s.bias])
    xfer_dest = torch.zeros(comfy.memory_management.vram_aligned_size([s.weight, s.bias]), dtype=torch.uint8)

    def lowvram(x):
        raise RuntimeError("LoRA patch failed")
    s.weight_lowvram_function = lowvram
    s._prefetch = {"signature": "new", "resident": False, "xfer_dest": xfer_dest, "cast_dest": None,
                   "cast_geometry": geometry, "needs_cast": False}
    with pytest.raises(RuntimeError, match="LoRA patch failed"):
        comfy.ops.resolve_cast_module_with_vbar(s, torch.float32, torch.device("cpu"), torch.float32, None, False)
    assert s._v_signature is None
