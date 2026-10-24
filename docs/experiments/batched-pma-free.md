# Batched PMA free experiment

This branch batches the PMA frees issued by eager eviction. Eager eviction
still selects and submits roots individually, but drains every selected root's
tracker, publishes `PMA_OWNED`, and then calls `nvUvmInterfacePmaFreePages`
once with the selected root addresses. The PMA read lock spans waits, state
publication, and the array free, preserving PMA eviction callback
synchronization. Eager roots are expected to have `is_zero == false`.

Generic single-root paths retain their existing mode and zero-page flags.
PMA-eviction callback paths remain unbatched.

## Limits

The experiment is limited to the existing eager batch array (16 roots). Linux
kernel builds and GPU/runtime validation are not possible in this macOS
environment; the extracted C regression test checks call cardinality, state
ordering, wait coverage, and address forwarding.

## Test

From the worktree root:

```sh
python3 tests/experiments/test_batched_pma_free.py
```

The test compiles its generated harness with `cc`, AddressSanitizer, and
UndefinedBehaviorSanitizer.
