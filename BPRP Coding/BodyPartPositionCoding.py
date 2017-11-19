# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 11:41:01 2017

@author: Hainan Chen
@Email: hn.chen@live.com
"""

import numpy as np
import pandas as pd
import pickle
from BVHSkeleton import *
from BodyPartPositionKits import *

import copy

'''
# get side view of the posture
# output: bones position from the side view (vertical comparision)
'''
def getSideView(testFrame, refFrame):
    refBonesPos = refFrame
    UpLegRef = testFrame['RightUpLeg'] - testFrame['LeftUpLeg']
    if np.abs(UpLegRef[0]) > np.abs(UpLegRef[2]):
        sideViewPose = getMapPos(testFrame, 'z') # map to xy plane
    else:
        sideViewPose = getMapPos(testFrame, 'x') # map to zy plane
        
    # find the lowest position of the side view posture
    lowestP = np.inf
    lowestBone = ""
    for k, v in sideViewPose.items():
        if "End" in k:
            continue
        if v[1] < lowestP: # lowest position of y
            lowestP = v[1]
            lowestBone = k
            
    testLowerHeight = sideViewPose['Hips'][1] - lowestP
    refLowerHeight = refBonesPos['Hips'][1] - refBonesPos['LeftFoot'][1]
    yAdjustHeight = testLowerHeight - refLowerHeight
    
    for k in sideViewPose.keys():
        sideViewPose[k][1] = sideViewPose[k][1] + yAdjustHeight
        
    return sideViewPose
    
'''
# get the top view
# compresing the 3d into 2d plane
# top view c: column, r: row
'''
def getTopView(testFrame):
    topViewPose = getMapPos(testFrame, 'y')
    return topViewPose

'''
# detect brent or straight depending on threshold value
# thresholdVal -0.94 (161 degree)
'''
def getPartPose(cosAngle, thresholdVal = -0.94):
    if cosAngle > thresholdVal:
        return "Bent"
    else:
        return "Straight"
    
    
'''
# coding the the posture which presented in testFrame
'''
def getPostureCoding(testFrame, refFrame):
    refBonesPos = refFrame
    
    sideViewPose = getSideView(testFrame, refFrame)
    
    topViewPose = getTopView(testFrame)
        
    postureCoding = {}
    poseCosVal = {}
    positionDetectionPart = ["Head", "LeftShoulder", "RightShoulder", "LeftHand", 
                             "RightHand","LeftFoot", "RightFoot"]
    
    for one in positionDetectionPart:
        vP = getSideViewPosition(refBonesPos, sideViewPose[one][1])
        tRowP = getTopRowViewPosition(topViewPose, topViewPose[one])
        tColP = getTopColViewPosition(topViewPose, topViewPose[one])
#        postureCoding['{0}Position'.format(one)] = {'frontVK':vP, 'topVCK':tColP, 'topVRK':tRowP}
        if 'Hand' in one:
            itemName = one.replace('Hand', 'Arm')
        elif 'Foot' in one:
            itemName = one.replace('Foot', 'LowerLimb')
        else:
            itemName = one
        postureCoding['{0}Position'.format(itemName)] = "{0}{1}{2}".format(vP, tColP, tRowP)
    
    '''
    # get the whole body learn trend 
    '''
    COMPosition = getCenterOfMass(testFrame)
    c = getTopColViewPosition(topViewPose, COMPosition)
    r = getTopRowViewPosition(topViewPose, COMPosition)
    postureCoding['BodyLeanTrend'] = "Le{0}{1}".format(c, r)
    
    '''
    # get the pose (brent or straight) for body parts
    '''
    leftArmV1 = testFrame['LeftArm'] - testFrame['LeftForeArm']
    leftArmV2 = testFrame['LeftHand'] - testFrame['LeftForeArm']
    cosLeftArm = getTwoVectorCos(leftArmV1, leftArmV2)
    postureCoding['LeftArmPose'] = getPartPose(cosLeftArm, -0.79)
    poseCosVal['LeftArmPose'] = cosLeftArm
    
    rightArmV1 = testFrame['RightArm'] - testFrame['RightForeArm']
    rightArmV2 = testFrame['RightHand'] - testFrame['RightForeArm']
    cosRightArm = getTwoVectorCos(rightArmV1, rightArmV2)
    postureCoding['RightArmPose'] = getPartPose(cosRightArm, -0.70)
    poseCosVal['RightArmPose'] = cosRightArm
    
    leftLegV1 = testFrame['LeftUpLeg'] - testFrame['LeftLeg']
    leftLegV2 = testFrame['LeftFoot'] - testFrame['LeftLeg']
    cosLeftLeg = getTwoVectorCos(leftLegV1, leftLegV2)  
    postureCoding['LeftLowerLimbPose'] = getPartPose(cosLeftLeg, -0.82)
    poseCosVal['LeftLowerLimbPose'] = cosLeftLeg
    
    rightLegV1 = testFrame['RightUpLeg'] - testFrame['RightLeg']
    rightLegV2 = testFrame['RightFoot'] - testFrame['RightLeg']
    cosRightLeg = getTwoVectorCos(rightLegV1, rightLegV2)
    postureCoding['RightLowerLimbPose'] = getPartPose(cosRightLeg, -0.72)
    poseCosVal['RightLowerLimbPose'] = cosRightLeg
    
    trunkV1 = testFrame['Spine3'] - testFrame['Spine2']
    trunkV2 = testFrame['Spine'] - testFrame['Spine1']
    cosTrunk = getTwoVectorCos(trunkV1, trunkV2)
    postureCoding['TrunkPose'] = getPartPose(cosTrunk, -0.986)
    poseCosVal['TrunkPose'] = cosTrunk

    return postureCoding, poseCosVal



if __name__ == "__main__":
    
    bvhObj = BVHParsing("sam001Char00.bvh")
    bvhObj.read()
    refBonesPos = {}
    getBodyPartABSPos(bvhObj.root, refBonesPos)
    refFrame = setHips2Origin(refBonesPos)
    
    with open('skProcessed4sam001.pkl', 'rb') as f:
        sk = pickle.load(f)
    bonesPos = sk.bonesFramesPos
    
    '''
    # conducting machine posture coding
    '''
#    normPostureCodingFrames = []
#    normPostureCodingCosVal = []
#    for i in np.arange(len(bonesPos)):
#        
#        frameIndex = i       
#        testFrame = bonesPos[frameIndex]
#        testFrame = setHips2Origin(testFrame)
#        testFrame = matchHipUplegs(testFrame)
#        
#        pCoding, poseCosVal = getPostureCoding(testFrame, refFrame)
#        normPostureCodingFrames.append(pCoding)
#        normPostureCodingCosVal.append(poseCosVal)
#        print(i)
#        
#    normPostureCodingFramesDf = pd.DataFrame(normPostureCodingFrames)
#    normPostureCodingCosValDf = pd.DataFrame(normPostureCodingCosVal)
#    
#    normPostureCodingFramesDf.to_hdf('normPostureCodingFrames.h5',
#                                     'normPostureCodingFramesDf', mode ='w')
#    
#    normPostureCodingCosValDf.to_hdf('normPostureCodingCosVal.h5',
#                                     'normPostureCodingCosValDf', mode='w')
    '''
    # end
    '''
    
    # save the data by HDFStore
    
    
    # save the data by pickle 
#    with open('normPostureCodingFramesDf.pkl', 'wb') as f:
#        pickle.dump(normPostureCodingFramesDf, f)
     
    
    '''
    # make the coding to be categorical data
    '''
#    pos = ['Bent', 'Straight']
#    vKeys = ['U', 'S', 'M', 'P', 'L']
#    cKeys = ['L', 'M', 'R']
#    rKeys = ['F', 'S', 'B']
#    codedPostureCodingFrames = []
#    for i in np.arange(normPostureCodingFramesDf.shape[0]):
#        print(i)
#        oneFrame = {}
#        for one in normPostureCodingFramesDf.columns:
#            val = normPostureCodingFramesDf.iloc[i, :][one]
#            if 'Pose' in one:
#                oneFrame[one] = pos.index(val)
#            elif 'Position' in one:
#                codingList = list(val)
#                oneFrame['{0}-v'.format(one)] = vKeys.index(codingList[0])
#                oneFrame['{0}-c'.format(one)] = cKeys.index(codingList[1])
#                oneFrame['{0}-r'.format(one)] = rKeys.index(codingList[2])
#            elif 'Trend' in one:
#                codingList = list(val)
#                oneFrame['{0}-c'.format(one)] = cKeys.index(codingList[2])
#                oneFrame['{0}-r'.format(one)] = rKeys.index(codingList[3])
#                
#        codedPostureCodingFrames.append(oneFrame)
#     codedPostureCodingFramesDf = pd.DataFrame(codedPostureCodingFrames)
#     with open('codedPostureCodingFramesDf.pkl', 'wb') as f:
#         pickle.dump(codedPostureCodingFramesDf, f)


    '''
    # image show
    '''
#    frameT2Image(frameIndex, sk, 'postureCSCoding.png') 
#
    
    
    '''
    # !!!
    # algorithm checking for the given frame of motion data
    '''
    frameIndex = 4282
    testFrame = bonesPos[frameIndex]
    testFrame = setHips2Origin(testFrame)
    testFrame = matchHipUplegs(testFrame)
    
    sideViewPose = getSideView(testFrame, refFrame)
    topViewPose = getTopView(testFrame)
  
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1,projection='3d')
    showOneFramePosture(refFrame, ax)
#    showOneFramePosture(testFrame, ax)
    
#    topViewP = topViewPose
#    partPosition = topViewPose['LeftFoot']
    
#    fig = plt.figure()
#    ax = fig.add_subplot(1,1,1,projection='3d')
#    showOneFramePosture(sideViewPose, ax)
#    showOneFramePosture(topViewPose, ax)
    
#    pCoding, poseCosVal = getPostureCoding(testFrame, refFrame)
    '''
    # end
    '''
