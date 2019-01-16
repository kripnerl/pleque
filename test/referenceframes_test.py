from pleque.spatran import affine
from pleque.spatran.reference_frames import ReferenceFrame
import numpy as np

trans2to1 = affine.Translate(np.array([1,0,0]))
trans3to1 = affine.Translate(np.array([0,1,0]))
transneworig = affine.Translate(np.array([0,1,1]))



##create a origin frame and test transformations:
frame2 = ReferenceFrame("frame2")
frame3 = ReferenceFrame("frame3", transform=trans3to1, parent=frame2)

vector = np.array([0, 0, 0])

print("from frame {0} : {1} transformed to parent frame {2}: {3}".format(frame3.name, vector,
                                                                         frame2.name, frame3.toparent(vector)))

print("from frame {0} : {1} transformed to child frame {2}: {3}".format(frame2.name, vector,
                                                                         frame3.name, frame2.tochild(frame3, vector)))

#add a parent node to frame2 to test changes of the parent node
frame1 = ReferenceFrame("frame1")

frame2.parent_add(frame1, trans2to1)

#test transformation to origin

print("from frame {0} : {1} transformed to origin frame {2}: {3}".format(frame3.name, vector,
                                                                        frame1.name, frame3.toorigin(vector)))

#test changing origin
neworigin = ReferenceFrame("New Origin")

frame2.parent_change(neworigin, transneworig)

print("from frame {0} : {1} transformed to origin frame {2}: {3}".format(frame3.name, vector,
                                                                        frame1.name, frame3.toorigin(vector)))

#check if loop can be created:

neworigin.parent_add(frame3, trans2to1)