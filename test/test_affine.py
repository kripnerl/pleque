from pleque.spatran import affine
import numpy as np

transmat = np.eye(4)
transmat[0,-1] = 4

#test tranlation
translate = affine.Affine(name="translate test", transform=transmat)

vector = np.array([0,0,0])
vector2 = np.array([[0,0,0],[0,0,1]]).T
print(translate * vector)

print(translate * vector2)
#test transformation inversion
print(~translate*(translate * vector2))

#test transformation adition
transmat = np.eye(4)
transmat[0,-1] = 4
transmat2 = np.eye(4)
transmat2[1,-1] = 2

translate = affine.Affine(name="translate test", transform=transmat)
translate2 = affine.Affine(name="translate test", transform=transmat2)

combined_translate = translate * translate2
print(combined_translate * vector2)

#test identity affine transform
affine.Identity(dim=3, name="test")

#test translation
vector = np.array([1, 0, 0])
translate = affine.Translate(np.array([1,0,0]))
print(translate * vector)

#test Scale
vector = np.array([1, 2, 3])
scale = affine.Scale(np.array([1, 2, 3]))
print(scale * vector)


#test Rotx
vector = np.array([0, 0, 1])
rotx = affine.Rotx(np.pi/2)
print(rotx * vector)

#test Roty
vector = np.array([1, 0, 0])
roty = affine.Roty(np.pi/2)
print(roty * vector)

#test Rotz
vector = np.array([1, 0, 0])
rotz = affine.Rotz(np.pi/2)
print(rotz * vector)