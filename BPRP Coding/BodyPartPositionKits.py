# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 15:21:37 2017

@author: Hainan Chen
@Email: hn.chen@live.com
"""

import numpy as np
import pandas as pd
import copy

import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D

'''
# show one frame posture through positions of bones (pf)
# pf dict {bonesName: (x, y, z)}: one frame of bones data
'''
def showOneFramePosture(pf, ax = None):
    if ax == None:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1,projection='3d')
        
    skStructure = [["Hips","RightUpLeg","RightLeg","RightFoot"],
                   ["Hips", "LeftUpLeg","LeftLeg","LeftFoot"],
                   ["Spine3", "RightShoulder","RightArm","RightForeArm","RightHand"],
                   ["Spine3", "LeftShoulder","LeftArm","LeftForeArm","LeftHand"],
                   ["Hips","Spine","Spine1","Spine2","Spine3", "Neck", "Head"]]
    
    
    for one in skStructure:
        oneSKLine = []
        flag = ""
        for oneBoneName in one:
            if "Left" in oneBoneName:
                flag = 'Left'
            elif "Right" in oneBoneName:
                flag = 'Right'
            oneSKLine.append(pf[oneBoneName])
            
            if oneBoneName == "Head":
                ax.scatter(pf[oneBoneName][0], pf[oneBoneName][1], pf[oneBoneName][2], c = 'orange',
                           marker = 's', s=100)
            
        skLineData = np.array(oneSKLine)
#        print(skLineData)
#        print(skLineData.shape)
        if flag == 'Left':
            color = "blue"
        elif flag == 'Right':
            color = "red"
        else:
            color = "green"        
        ax.plot(skLineData[:, 0], skLineData[:, 1], skLineData[:, 2], c = color)
        
        
        
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
 
'''
# get reference T pose bones position data through calculating the 
# accumulated offset
# root: the root node of the bones nodes tree, established through bvh file
# refBonesBos: Output, the results of the bones initial posutre (T pose)
'''
def getBodyPartABSPos(root, refBonesPos, preNode = None):
    boneName = root.name
    
    root.offset = np.array(root.offset)
    if preNode != None:
        root.offset = root.offset + preNode.offset
        
    refBonesPos[boneName] = root.offset
    
    if len(root.children) > 0:
        for one in root.children:
            getBodyPartABSPos(one,  refBonesPos, root)
            
'''
# put the bones positions to be the relative position to the hips
# and move the hips to the origin point
# bonesPosDict: dict{bone name: [x, y, z]}
'''
def setHips2Origin(bonesPosDict):
    relHipsPos = {}
    for k, v in bonesPosDict.items():
        relV = v - bonesPosDict['Hips']
        relHipsPos[k] = relV  
    return relHipsPos

'''
# make the left and right upleg bones and the hips matched
# make sure the offset among different postures to be smallest
# normal vector of v1(Hips->RightUpLeg) to v2(Hips->LeftUpLeg) denotes the front
'''
def matchHipUplegs(bonesPosDict):

    UpLegsV = bonesPosDict['LeftUpLeg'] - bonesPosDict['RightUpLeg']
    
    rM = getRWithYToXAxisM(UpLegsV)   
    
    rBonesPosDict = copy.deepcopy(bonesPosDict)
    
    for k, v in rBonesPosDict.items():
        vM = np.matrix(v)
        vR = np.matmul(vM, rM) # rotated position
        rBonesPosDict[k] = np.asarray(vR[0,:])[0]
        
    return rBonesPosDict

'''
# Get the rotation matrix let posture rotating with y axis
# and make the face to positive z axis
# vL2R is the vector with start with RightUpLeg and ended with LeftUpLeg
# looks stupid, but it works!
'''
def getRWithYToXAxisM(vL2R):
    
    x, y, z = vL2R[0],vL2R[1],vL2R[2]
    if x == 0 and z > 0:
        s = 0
    elif x == 0 and z < 0:
        s = np.pi
    elif z == 0 and x > 0:
        s = np.pi / 2
    elif z == 0 and x < 0:
        s = 3/2 * np.pi
    elif x > 0 and z > 0:
        s = 2 * np.pi - np.arctan(z/x)
    elif x > 0 and z < 0:
        s = -np.arctan(z/x)
    elif x < 0 and z < 0:
        s = np.pi / 2 + np.arctan(x/z)
    elif x < 0 and z > 0:
        s = (3/2) * np.pi + np.arctan(x/z)
    
    # around y axis positive rotation, s is the pitch angle
    # it's the negative direction of right-handed helix
    rM = np.matrix([[np.cos(s), 0, np.sin(s)],
                    [0,         1,         0],
                    [-np.sin(s), 0, np.cos(s)]])
    return rM

'''
# compress the 3d skeleton data into 2d through seting
# the value of the given axis to be zeros. 
# get the results for the positions mapped into denoted plane
# d x, y, or z denote the axis which should be eliminated
'''
def getMapPos(pos, d):
    axis = ['x', 'y', 'z']
    i = axis.index(d)
    mapPos = copy.deepcopy(pos)
    
    for k in mapPos.keys():
        mapPos[k][i] = 0
        
    return mapPos


'''
# first get the vertical position of the body part from the side view. 
# get the frontVK value.
# ref: the initial T pose body part bones position
# yPos: the y position of the given body part.
# the y axis is considered as the vertical direction.
# Negative y direction is the same direction of G force.
'''
def getSideViewPosition(ref, yPos):
    vP = ""
    shoulderP = (ref['LeftShoulder'][1] + ref['RightShoulder'][1]) / 2
    pocketP = (ref['LeftUpLeg'][1] + ref['RightUpLeg'][1]) / 2
    if yPos > shoulderP + 10:
        vP = "U"
    elif yPos > shoulderP - 10 and yPos <= shoulderP + 10:
        vP = "S"
    elif yPos > pocketP + 10 and yPos <= shoulderP - 10:
        vP = "M"
    elif yPos > pocketP - 10 and yPos <= pocketP + 10:
        vP = "P"
    else:
        vP = "L"
        
    return vP

'''
# second, get the body position in the xz plane from the top view.
# topViewP: bonesPosDict which has been compressed into xz plane.
# partPosition: the position of the given body part bone
# results: the relative position among left, middle, right
'''
def getTopColViewPosition(topViewP, partPosition):
    
    refUpLegs = topViewP['RightUpLeg'] - topViewP['LeftUpLeg']
    testPart = partPosition - topViewP['LeftUpLeg']
    
    refNorm = np.linalg.norm(refUpLegs)
    testNorm = np.linalg.norm(testPart)
    
    cosRefTest = np.dot(refUpLegs, testPart) / (refNorm * testNorm)
    mapLength = testNorm * cosRefTest
    
    tColP = ""
    if cosRefTest >= -0.5 and mapLength <= refNorm:
        tColP = "M"
    elif cosRefTest >= -0.5 and mapLength > refNorm:
        tColP = "R"
    else:
        tColP = "L"
        
    return tColP

'''
# third, get the body position in the xz plane from the top view.
# topViewP: bonesPosDict which has been compressed into xz plane.
# partPosition: the the position of the test part
# results: the relative position among front, side, back 
'''
def getTopRowViewPosition(topViewP, partPosition):
    
    refUpLegs = topViewP['RightUpLeg'] - topViewP['LeftUpLeg']
    testPart = partPosition - topViewP['LeftUpLeg'] #topViewP[partName] - topViewP['LeftUpLeg']
    
    refNorm = np.linalg.norm(refUpLegs)
    testNorm = np.linalg.norm(testPart)
    
    cosRefTest = np.dot(refUpLegs, testPart) / (refNorm * testNorm)
    dist = testNorm * np.sqrt(1 - cosRefTest ** 2)
    
    A = topViewP['LeftUpLeg']
    B = topViewP['RightUpLeg']
    C = partPosition
    # right hand rule !!!
    # z in 3d to be x in 2d
    # x in 3d to be y in 2d
    x1, x2, x3 = A[2], B[2], C[2]
    y1, y2, y3 = A[0], B[0], C[0]
    s = (x1 - x3) * (y2 - y3) - (y1 - y3) * (x2 - x3)
    
    tRowP = ""
    if dist <= 5:
        tRowP = 'S'
    elif s > 0:
        tRowP = "F"
    elif s < 0:
        tRowP = "B"
        
    return tRowP

def getTwoVectorCos(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    
    v1Norm = np.linalg.norm(v1)
    v2Norm = np.linalg.norm(v2)
    
    cosRefTest = np.dot(v1, v2) / (v1Norm * v2Norm)
    return cosRefTest
    
'''
# Estimate the center of mass by segmentation method
# pf: dict{boneName:[x,y,z]} the bones position data
# This can only be a rough estimation of COM
# Ref: Design A Model for Human Body to Determine the Center of Gravity
'''
def getCenterOfMass(pf):
    SegW = {'Head':8.26, 'Trunk':46.84, 'LeftArm':3.25, 'RightArm':3.25,
            'LeftForeArm':1.87, 'RightForeArm':1.87, 'LeftHand':0.65, 'RightHand':0.65,
            'LeftUpLeg':10.5, 'RightUpLeg':10.5, 'LeftLeg':4.75, 'RightLeg':4.75,
            'LeftFoot':1.43, 'RightFoot':1.43}
    
    COMPosition = None
    i = 0
    for k, v in SegW.items():
        
        if k == 'Trunk':
            segPos = (np.array(pf['Spine']) + np.array(pf['Spine']) + np.array(pf['Spine'])
                        + np.array(pf['Spine']) + np.array(pf['Hips'])) / 5
        else:
            segPos = np.array(pf[k])
        
        if i == 0:
            COMPosition = segPos * v
        else:
            COMPosition = COMPosition + segPos * v
        i = i + 1
    
    return COMPosition / 100

if __name__ == "__main__":
    print("Kits for body part position coding")
    