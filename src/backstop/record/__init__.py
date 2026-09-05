"""Recording package.

`loop` imports numpy at module scope, which arrives through the `sim` extra.
`timing` deliberately imports nothing outside the standard library, so that the
corpus-sizing arithmetic it produces can be tested on the CPU CI job where the
sim extra is not installed.

Re-exporting `loop` eagerly here defeats that: `import backstop.record.timing`
runs this file first, so it would drag numpy in and the timer's tests would error
out of the cheap job. Hence the lazy re-export -- `from backstop.record import
create_dataset` still works, but only pays for numpy when someone asks for it.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from backstop.record.loop import create_dataset, run_episodes

__all__ = ["create_dataset", "run_episodes"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from backstop.record import loop  # noqa: PLC0415

        return getattr(loop, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
