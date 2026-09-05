"""RetryNow — AI Revenue Recovery Engine.

RetryNow predicts, for every failed payment, the single best recovery action:
retry soon, retry later, switch instrument, send a link, or give up.

Sub-packages are imported lazily by their consumers (not here) so that a single
module can always be imported without triggering the whole dependency tree.
"""

__version__ = "1.0.0"