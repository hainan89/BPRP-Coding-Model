# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 11:06:26 2017

@author: luo's lab
"""
import numpy as np
from numpy import array, dot
import copy
import cgkit.bvh
from math import radians, cos, sin

class Joint():
    
    def __init__(self, name):
        self.name = name
        self.children = []
        
        self.channels = []
        
        self.hasparent = 0
        self.parent = None
        
        # offset from bvh node info
        self.strans = np.array([0,0,0])
        
        # matrix value for strans
        self.stransmat = None
        
        # motion frame index based translation and position value
        self.trtr = {}
        self.worldpos = {}
        
    def addChild(self, childJoint):
        self.children.append(childJoint)
        childJoint.hasparent = 1
        childJoint.parent = self
        
class Skeleton():
    
    def __init__(self, filename):
        
        bvhObj = BVHParsing(filename)
        bvhObj.read()
        print('read done')
        rootJoint = bvhObj.processBVHNode(bvhObj.root, None)

        self.rootJoint = rootJoint
        self.framesValues = bvhObj.framesValues
        self.frames = bvhObj.frames
        self.dt = bvhObj.dt
        self.bonesFramesPos = {}
    
    def processTFrameMotion(self, t):
        oneFrameValue = self.framesValues[t]
        bonesPos = {}
        self.processOneFrameMotion(oneFrameValue, self.rootJoint, t, bonesPos)
        self.bonesFramesPos[t] = bonesPos
        
    # frameSeg is the segment of the frame value for the unprocessed joints
    
    def process4EndSite(self, joint, t):
        parentJoint = joint.parent
        joint.worldpos[t] = parentJoint.worldpos[t] + joint.strans
        return None
    
    '''
    # frameSegVal is one frame motion data, start from the corresponding joint.
    # 
    '''
    def processOneFrameMotion(self, keyframe, joint, t, bonesPos):
        
        counter = 0
        dotrans = 0
        dorot = 0
        drotmat = array([ [1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.] ])
        
        for channel in joint.channels:
            keyval = keyframe[counter]
            if(channel == "Xposition"):
                dotrans = 1
                xpos = keyval
            elif(channel == "Yposition"):
                dotrans = 1
                ypos = keyval
            elif(channel == "Zposition"):
                dotrans = 1
                zpos = keyval
            elif(channel == "Xrotation"):
                dorot = 1
                xrot = keyval
                theta = radians(xrot)
                mycos = cos(theta)
                mysin = sin(theta)
                drotmat2 = array([ [1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.], 
                                    [0.,0.,0.,1.] ])
                drotmat2[1,1] = mycos
                drotmat2[1,2] = -mysin
                drotmat2[2,1] = mysin
                drotmat2[2,2] = mycos
                drotmat = dot(drotmat, drotmat2)
            elif(channel == "Yrotation"):
                dorot = 1
                yrot = keyval
                theta = radians(yrot)
                mycos = cos(theta)
                mysin = sin(theta)
                drotmat2 = array([ [1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],
                                    [0.,0.,0.,1.] ])
                drotmat2[0,0] = mycos
                drotmat2[0,2] = mysin
                drotmat2[2,0] = -mysin
                drotmat2[2,2] = mycos
                drotmat = dot(drotmat, drotmat2)
            elif(channel == "Zrotation"):
                dorot = 1
                zrot = keyval
                theta = radians(zrot)
                mycos = cos(theta)
                mysin = sin(theta)
                drotmat2 = array([ [1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],
                                    [0.,0.,0.,1.] ])
                drotmat2[0,0] = mycos
                drotmat2[0,1] = -mysin
                drotmat2[1,0] = mysin
                drotmat2[1,1] = mycos
                drotmat = dot(drotmat, drotmat2)
            else:
                print ("Fatal error in process_bvhkeyframe: illegal channel name ", channel)
                return(0)
            counter += 1
    # End "for channel..."

        if dotrans:  # If we are the hips...
            # Build a translation matrix for this keyframe
            dtransmat = array([ [1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],
                                [0.,0.,0.,1.] ])
            dtransmat[0,3] = xpos
            dtransmat[1,3] = ypos
            dtransmat[2,3] = zpos
        if joint.hasparent:  # Not hips
            parent_trtr = joint.parent.trtr[t]  # Dictionary-based rewrite
            localtoworld = dot(parent_trtr,joint.stransmat)
        else:  # Hips
            localtoworld = dot(joint.stransmat,dtransmat)
        
        trtr = dot(localtoworld,drotmat)
        
        joint.trtr[t] = trtr  # New dictionary-based approach    
        
        
        worldpos = array([ localtoworld[0,3],localtoworld[1,3],
                            localtoworld[2,3]])
    ##  joint.worldpos.append(worldpos)
        joint.worldpos[t] = worldpos  # Dictionary-based approach
        
#        print(joint.name, worldpos)
        
        bonesPos[joint.name] = worldpos
        
        newkeyframe = keyframe[counter:]  # Slices from counter+1 to end
        for child in joint.children:
            newkeyframe = self.processOneFrameMotion(newkeyframe, child, t, bonesPos)
            if(newkeyframe == 0):  # If retval = 0
                print ("Passing up fatal error in process_bvhkeyframe")
                return None
        return newkeyframe
        
        
class BVHParsing(cgkit.bvh.BVHReader):
    
    def __init__(self, filename):
        cgkit.bvh.BVHReader.__init__(self, filename)
        self.frames = 0
        self.dt = 0.0
        self.framesValues = []
        self.root = None
        
    def onHierarchy(self, root):
#        print('hello root')
        self.root = root
        
    def onMotion(self, frames, dt):
#        print('nice dt')
        self.frames = frames
        self.dt = dt

    def onFrame(self, values):
#        print('nice values')
        self.framesValues.append(values)     
        
    def processBVHNode(self, node, parentName):
        name = node.name
        if (name == "End Site") or (name == "end site"):
            name = parentName + "End"
            
        joint1 = Joint(name)
        joint1.channels = copy.deepcopy(node.channels)
        joint1.strans = copy.deepcopy(node.offset)
        
        joint1.stransmat = np.array([[1,0,0,node.offset[0]],
                                     [0,1,0,node.offset[1]],
                                     [0,0,1,node.offset[2]],
                                     [0,0,0,1]])
        
        for child in node.children:
            joint2 = self.processBVHNode(child, name)
            joint1.addChild(joint2)
            
        # root joint of the skeleton
        return joint1
    
'''
##################
'''    

def getBodyPart(root, t, bodyPart = [], bonesList = [], preNode = None):
    
    if preNode != None:
#        print(root.name, preNode.name)
        bodyPart.append(list(zip(preNode.worldpos[t], root.worldpos[t])))
        bonesList.append([preNode.name, root.name])
        
    if len(root.children) > 0:
        for one in root.children:
            getBodyPart(one, t, bodyPart, bonesList, root)

def updateOffset(root, preNode = None):
    
    if preNode != None:
        root.offset = preNode.offset + np.array(root.offset)
    else:
        root.offset = np.array(root.offset)
    if len(root.children) > 0:
        for one in root.children:
            updateOffset(one, root)
    return
    
def getDefaulPos(root, bodyPart = [], bonesList = [], preNode = None):
    if preNode != None:
#        print(root.name, preNode.name)
        bodyPart.append(list(zip(preNode.offset, root.offset)))
        bonesList.append([preNode.name, root.name])
        
    if len(root.children) > 0:
        for one in root.children:
            getDefaulPos(one, bodyPart, bonesList, root)
 
'''
# draw the image for frame t
# t: frame index, sk: processed skeleton data
'''
def frameT2Image(t, sk, storageFileName):
    import matplotlib.pylab as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    bodyPart = []
    bonesList = []
    getBodyPart(sk.rootJoint, t, bodyPart, bonesList)
    
    # fig left
    fig = plt.figure(figsize=(12, 4))
    axI = 0
    viewParameter = [{'azim':0, 'elev':10},{'azim':0, 'elev':90},{'azim':90, 'elev':10}]
    for i in np.arange(len(viewParameter)):  #in [0, 45, 90]:
        axI = axI + 1
        ax = fig.add_subplot(1, 3, axI, projection='3d')
        
        for one in list(zip(bodyPart, bonesList)):
            if "Left" in one[1][0] and "Left" in one[1][1]:
                color = "blue"
            elif "Right" in one[1][0] and "Right" in one[1][1]:
                color = "red"
            else:
                color = "green"
                
            ax.plot(one[0][0],np.array(one[0][2]) * -1, one[0][1], color) # y to z
            
            if "HeadEnd" in one[1][0]:
                ax.scatter(one[0][0][0],-1*one[0][2][0],one[0][1][0],
                           marker='D',s=100, color="orange")
            elif "HeadEnd" in one[1][1]:
                ax.scatter(one[0][0][1],-1*one[0][2][1],one[0][1][1],
                           marker='D',s=100, color="orange")
        
        bodyPos = sk.rootJoint.worldpos[t]
        ax.set_xlim((bodyPos[0] - 100, bodyPos[0] + 100))
        ax.set_ylim((-1*bodyPos[2] - 100, -1*bodyPos[2] + 100))
        ax.set_zlim((bodyPos[1] - 100, bodyPos[1] + 100))
        
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.zaxis.set_ticklabels([])
        
#        ax.azim = azim #‘azim’ stores the azimuth angle in the x,y plane.
#        ax.elev = 10 #‘elev’ stores the elevation angle in the z plane
        ax.azim = viewParameter[i]['azim']
        ax.elev = viewParameter[i]['elev']
        
        ax.set_title('Azim :{0}, Elev: {1}'.format(viewParameter[i]['azim'],
                     viewParameter[i]['elev']))
        
    plt.suptitle('Views for Posture (Blue: Left, Red: Right)')   
    plt.tight_layout()
#    plt.savefig('./sam001/{0}.png'.format(t), dpi=75)
    plt.savefig(storageFileName, dpi=75)
    plt.close(fig)
    
if __name__ == "__main__":
    print('Hello BVH process')
    sk = Skeleton('sam001Char00.bvh')
    
#    showJoint = ['']
#    testSK = BVHParsing('sam001Char00.bvh')
#    testSK.read()
#    updateOffset(testSK.root)
#    bodyPartD = []
#    bonesListD = []
#    getDefaulPos(testSK.root, bodyPartD, bonesListD)

    import pandas as pd
    frameVpd = pd.DataFrame(sk.framesValues)
    
    t = 7463
    sk.processTFrameMotion(t)
    storageFileName = './sam001/{0}.png'.format(t)
    frameT2Image(t, sk, storageFileName)
      
    
    
        
        
        
        
        
        
        
        