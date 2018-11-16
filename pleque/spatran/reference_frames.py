class ReferenceFrame():

    def __init__(self, name, transform=None, parent=None, toparent=True):
        self._name = name
        self._child = {}
        self._transform_toparent = None  # from self to parent
        self._transform_toorigin = None  # from self to origin

        self._init_tranforms(transform, parent, toparent)

        # todo:remove child
        # todo:specifying of the transformation to the parent frame by more ways (IMAS style, some engineering styles etc.)
        # todo:introduce checks to prevent creation of closed loops in the tree

    def _init_tranforms(self, transform, parent, toparent):

        if transform is None or parent is None:
            self._transform_toparent = None
            self._parent = None
        else:
            if not toparent:
                transform = ~transform

            parent._child_add(self)
            self._check_frameloops(parent)  # check if there are some frame loops
            self._parent = parent
            self._transform_toparent = transform

            if not parent._transform_toorigin is None:
                self._transform_toorigin = (parent._transform_toorigin * transform)
            else:
                self._transform_toorigin = transform

    def _child_add(self, child):
        if child in self._child.values():
            raise Exception("Child already exists")
        else:
            for i in list(self._child.values()):
                if i is self:
                    raise Exception("Possible creation of a closed loop of reference frames. Child not added")
            self._child[child._name] = child

    def _child_remove(self, child):

        if child in self._child.values():
            del self._child[child._name]
        elif child._name in self._child.keys():
            del self._child[child]

    def _check_frameloops(self, frame):
        for i in list(self._child.values()):
            if i is frame:
                raise Exception("Possible loop in frame tree")
            i._check_frameloops(frame)

    def _transform_toorigin_changed(self):

        if self._parent._transform_toorigin is not None:
            self._transform_toorigin = self._parent._transform_toorigin * self._transform_toparent
        else:
            self._transform_toorigin = self._transform_toparent

        for i in list(self._child.values()):
            i._transform_toorigin_changed()

    def _parent_change(self, transform):
        self._transform_toparent = transform
        pass

    def parent_add(self, parent, transform, toparent=True):
        self._check_frameloops(parent)
        self._init_tranforms(transform, parent, toparent)
        self._transform_toorigin_changed()

    def parent_change(self, parent=None, transform=None, toparent=True):

        if parent is None:
            if self._parent is not None:
                parent = self._parent
            else:
                raise Exception("Parent of the frame has to be specified")
        else:  # remove self from child list of obsolete parent
            self._parent._child_remove(self)

        if transform is None:
            if self._transform_toparent is not None:
                transform = self._transform_toparent
            else:
                raise Exception("Transformation to parent has to be specified")

        self._init_tranforms(transform, parent, toparent)
        self._transform_toorigin_changed()

    def parent_remove(self):
        if self._parent is not None:
            self._parent._child_remove(self)

        self._transform_toparent = None
        self._parent = None
        self._transform_toorigin = None

    def toparent(self, vector):
        if self._transform_toparent is not None:
            return self._transform_toparent * vector
        else:
            raise Exception("No parent frame specified")

    def toorigin(self, vector):
        if self._transform_toorigin is not None:
            return self._transform_toorigin * vector

    def tochild(self, child, vector):

        if isinstance(child, ReferenceFrame):
            return ~child._transform_toparent * vector
        elif isinstance(child, str):
            return ~self._child[child.name].transform_parent * vector

    @property
    def name(self):
        return self._name

    @property
    def transform_toparent(self):
        return self._transform_toparent

    @property
    def transform_toorigin(self):
        return self._transform_toorigin
