import numpy as np
import matplotlib.pyplot as plt
import GPHS441_plates as gp

#%matplotlib notebook

lonlatAuckland = [174.7850,-37.0082]
lonlatWellington = [174.8076,-41.3276]

pAuckland = gp.positionVector(lonlatAuckland)
pWellington = gp.positionVector(lonlatWellington)

print('Auckland =',pAuckland,'; Wellington =',pWellington)

midpoint = (pAuckland + pWellington) / gp.length(pAuckland + pWellington)
lonlatMidpoint = gp.lonlat(midpoint)
print('Midpoint at ', lonlatMidpoint)

angle = np.arccos(np.dot(pAuckland,pWellington))
print('Distance = ',np.degrees(angle),'degrees = ',angle*gp.rEarth,'km')

pole = np.cross(pAuckland,pWellington) / np.sin(angle)
ll = gp.lonlat(pole)
print('Pole to great circle: position vector = ',pole,'; [lon,lat] = [{0:3.3f},{1:2.3f}]'.format(ll[0],ll[1]))

h = angle * pole
rotationMatrix = gp.rotationMatrix(h)
rotatedAuckland = np.dot(rotationMatrix,pAuckland)
print('pWellington = ',pWellington,'; rotatedAuckland = ',rotatedAuckland)

fig,ax = gp.mapSetup([170,180,-42,-34],"Auckland to Wellington, with points between")

# plot Auckland, Wellington, with line between
ax.plot([lonlatAuckland[0], lonlatWellington[0]], [lonlatAuckland[1], lonlatWellington[1]],
         color='blue', linewidth=1, marker='o', markersize=10)

# Use rotation to plot 0.1,0.2...0.9 of trip along great circle path (9 points).
print('proportion domain =', np.linspace(0.1,0.9,9))
for proportion in np.linspace(0.1,0.9,9):
    rotationMatrix = gp.rotationMatrix(proportion * h)
    rotatedAuckland = np.dot(rotationMatrix,pAuckland) 
    ax.plot(gp.lonlat(rotatedAuckland)[0], gp.lonlat(rotatedAuckland)[1],
            color='darkred', markersize=5, marker='o')

# Now consider a semi-circular (small circle) path with lots of points around the midpoint
# Rotation parameters: 180 degrees is pi radians, and the path is around the midpoint
hNew = np.pi * midpoint 
# Use a list [] to store each point along the path. Auckland is the start point
path = list()
path.append(lonlatAuckland)
# We could also instantiate the list in one line with: path = [lonlatAuckland]
n = 40
for i in range(1,n):
    rotationMatrix = gp.rotationMatrix(i/n * hNew)
    rotatedPositionVector = np.dot(rotationMatrix,pAuckland)
    path.append(gp.lonlat(rotatedPositionVector))

# Convert the path list to an array and transpose, to give better shape for plotting
pathArray = np.transpose(np.asarray(path)) # could also have used = np.asarray(path).T
print('pathArray \n',pathArray)
    
ax.plot(pathArray[0],pathArray[1], color='darkblue', linewidth=2)

plt.show()
