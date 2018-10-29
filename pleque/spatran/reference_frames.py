from pleque.spatran import affine


class ReferenceFrame():

    def __init__(self, name, transform=None, parent=None, toparent = True):
        self._name = name
        self._child = {}
        self._transform_toparent = None #from self to parent
        self._transform_toorigin = None #from self to origin

        self._init_tranforms(transform, parent, toparent)

        # todo:initiate transformations
        # todo:add child
        # todo:remove child
        # todo:add parent
        # todo:changed parent or anything in the chain above
        # todo:initiate with parent
        # todo:initiate without parent
        # todo:changed child
        # todo:Adding a child has to be initiated by the child
        # todo: handle vector transformation to parents, origins, children

    def _init_tranforms(self, transform, parent, toparent):
        # todo: add self as child of parent, if parent is passed
        # todo: add transformation to parent, if parent is passed
        # todo: child

        if transform is None or isinstance(transform, affine.Identity) or parent is None:
            self._transform_toparent = None
            self._parent = None
        else:
            if not toparent:
                transform = ~transform

            self._transform_toparent = transform
            parent._child_add(self)

            if not parent._transform_toorigin is None:
                self._transform_toorigin = (parent._transform_toorigin * transform)
            else:
                self._transform_toorigin = transform

    def _child_add(self, child):
        if child in self._child.values():
            raise Exception("Child already exists")
        else:
            self._child[child._name] = child

    def _child_remove(self, child):

        if child in self._child.values():
            del self._child[child._name]
        elif child._name in self._child.keys():
            del self._child[child]

    def _transform_toorigin_changed(self):
        self._transform_toorigin = self._parent._transform_toorigin * self._transform_toparent
        for i in self._child.values():
            i._tranform_toorigin_changed()

    def _parent_change(self, transform):
        self._transform_toparent = transform
        pass

    def add_parent(self, parent):
        self

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