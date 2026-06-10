import functools
import inspect
import warnings

string_types = (type(b''), type(u''))


def deprecated(reason):
    """
    This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used.
    """

    if isinstance(reason, string_types):

        # The @deprecated is used with a 'reason'.
        #
        # .. code-block:: python
        #
        #    @deprecated("please, use another function")
        #    def old_function(x, y):
        #      pass

        def decorator(func1):

            if inspect.isclass(func1):
                fmt1 = "Call to deprecated class {name} ({reason})."
            else:
                fmt1 = "Call to deprecated function {name} ({reason})."

            @functools.wraps(func1)
            def new_func1(*args, **kwargs):
                warnings.simplefilter('always', DeprecationWarning)
                warnings.warn(
                    fmt1.format(name=func1.__name__, reason=reason),
                    category=DeprecationWarning,
                    stacklevel=2
                )
                warnings.simplefilter('default', DeprecationWarning)
                return func1(*args, **kwargs)

            return new_func1

        return decorator

    elif inspect.isclass(reason) or inspect.isfunction(reason):

        # The @deprecated is used without any 'reason'.
        #
        # .. code-block:: python
        #
        #    @deprecated
        #    def old_function(x, y):
        #      pass

        func2 = reason

        if inspect.isclass(func2):
            fmt2 = "Call to deprecated class {name}."
        else:
            fmt2 = "Call to deprecated function {name}."

        @functools.wraps(func2)
        def new_func2(*args, **kwargs):
            warnings.simplefilter('always', DeprecationWarning)
            warnings.warn(
                fmt2.format(name=func2.__name__),
                category=DeprecationWarning,
                stacklevel=2
            )
            warnings.simplefilter('default', DeprecationWarning)
            return func2(*args, **kwargs)

        return new_func2

    else:
        raise TypeError(repr(type(reason)))


def append_to_doc(*snippets):
    """
    Append shared documentation snippets to the docstring of the decorated
    function (or of a function passed to the returned decorator).

    The original docstring is dedented with :func:`inspect.getdoc`, so the
    appended snippets can be written flush-left and the result still renders
    correctly with Sphinx autodoc. The function object is modified in place;
    no wrapper is created, so there is no runtime overhead.
    """

    text = '\n\n'.join(snippet.strip('\n') for snippet in snippets)

    def decorator(func):
        doc = inspect.getdoc(func)
        if doc:
            func.__doc__ = doc.rstrip() + '\n\n' + text + '\n'
        else:
            # The leading newline keeps the first snippet line out of the
            # docstring-dedent logic of autodoc/inspect.cleandoc, which would
            # otherwise strip the indentation of directive continuation lines.
            func.__doc__ = '\n' + text + '\n'
        return func

    return decorator


def scalar_function(func):
    """
    Serves to register class functions of Equilibrium class as scalar functions.
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        return func(self, *args, **kwargs)

    wrapper._scalar_function = True

    return wrapper


def ordered_path_scalar_function(func):
    """
    Register scalar functions that require ordered non-grid coordinates.

    These functions use geometry derived from neighbouring coordinate points
    (for example a wall or target surface normal), so rectangular grids and
    mesh-shaped point arrays are not meaningful inputs.
    """

    wrapper = scalar_function(func)
    wrapper._requires_ordered_path = True

    return wrapper


def vector_function(ndim):
    """
    Serves to register class functions of Equilibrium class as vector functions.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            return func(self, *args, **kwargs)
        wrapper._vector_function = True
        wrapper._vector_ndim = ndim
        return wrapper

    return decorator
