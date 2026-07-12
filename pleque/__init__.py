import logging

__version__ = '0.0.10'

logging.getLogger(__name__).addHandler(logging.NullHandler())

_LOG_HANDLER_ATTR = "_pleque_handler"


def set_log_level(level=logging.INFO):
    """
    Convenience helper to make PLEQUE log messages visible.

    Sets the level of the ``pleque`` logger and attaches a single
    :class:`logging.StreamHandler` to it (idempotent — repeated calls reuse
    the same handler). For full control configure the ``pleque`` logger
    directly with the standard :mod:`logging` machinery instead.

    :param level: logging level, e.g. ``logging.DEBUG`` (default ``logging.INFO``)
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    handler = next((h for h in logger.handlers if getattr(h, _LOG_HANDLER_ATTR, False)), None)
    if handler is None:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s: %(message)s"))
        setattr(handler, _LOG_HANDLER_ATTR, True)
        logger.addHandler(handler)
    handler.setLevel(level)


from pleque.core import *  # noqa: E402
