import numpy as np
cimport cython
from libc.stdio cimport printf
from libc.math cimport atan2
cdef extern from "math.h":
    double sqrt(double m)


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef doAgentSense_lowrange(data):
    """
     Sensor model is <aNE, aNW, aSW, aSE, pNE, pNE, pSW, pSE>
     Where a means (other) agent, p means poi, and the rest are the quadrants
    """
    cdef int number_agents = data['Number of Agents']
    cdef int number_pois = data['Number of POIs'] 
    cdef double minDistanceSqr = data["Minimum Distance"] ** 2
    cdef double[:, :] agentPositionCol = data["Agent Positions"]
    cdef double[:] poiValueCol = data['Poi Values']
    cdef double[:, :] poiPositionCol = data["Poi Positions"]
    cdef double[:, :] orientationCol = data["Agent Orientations"]
    npObservationCol = np.zeros((number_agents, 8), dtype = np.float64)
    cdef double[:, :] observationCol = npObservationCol
    
    cdef int agentIndex, otherAgentIndex, poiIndex, obsIndex
    cdef double globalFrameSeparation0, globalFrameSeparation1
    cdef double agentFrameSeparation0, agentFrameSeparation1

    cdef double distanceSqr
    cdef double lowrange=7.5*7.5
    
    
    for agentIndex in range(number_agents):

        # calculate observation values due to other agents
        for otherAgentIndex in range(number_agents):
            
            # agents do not sense self (ergo skip self comparison)
            if agentIndex == otherAgentIndex:
                continue
                
            # Get global separation vector between the two agents    
            globalFrameSeparation0 = agentPositionCol[otherAgentIndex,0] - agentPositionCol[agentIndex,0]
            globalFrameSeparation1 = agentPositionCol[otherAgentIndex,1] - agentPositionCol[agentIndex,1]
            
            # Translate separation to agent frame using inverse rotation matrix
            agentFrameSeparation0 = orientationCol[agentIndex, 0] * globalFrameSeparation0 + orientationCol[agentIndex, 1] * globalFrameSeparation1 
            agentFrameSeparation1 = orientationCol[agentIndex, 0] * globalFrameSeparation1 - orientationCol[agentIndex, 1] * globalFrameSeparation0 
            distanceSqr = agentFrameSeparation0 * agentFrameSeparation0 + agentFrameSeparation1 * agentFrameSeparation1
            
            # By bounding distance value we implicitly bound sensor values
            if distanceSqr < minDistanceSqr:
                distanceSqr = minDistanceSqr

            if distanceSqr < lowrange:
            
                # other is east of agent
                if agentFrameSeparation0 > 0:
                    # other is north-east of agent
                    if agentFrameSeparation1 > 0:
                        observationCol[agentIndex,0] += 1.0 / distanceSqr
                    else: # other is south-east of agent
                        observationCol[agentIndex,3] += 1.0  / distanceSqr
                else:  # other is west of agent
                    # other is north-west of agent
                    if agentFrameSeparation1 > 0:
                        observationCol[agentIndex,1] += 1.0  / distanceSqr
                    else:  # other is south-west of agent
                        observationCol[agentIndex,2] += 1.0  / distanceSqr



        # calculate observation values due to pois
        for poiIndex in range(number_pois):
            
            # Get global separation vector between the two agents    
            globalFrameSeparation0 = poiPositionCol[poiIndex,0] - agentPositionCol[agentIndex,0]
            globalFrameSeparation1 = poiPositionCol[poiIndex,1] - agentPositionCol[agentIndex,1]
            
            # Translate separation to agent frame unp.sing inverse rotation matrix
            agentFrameSeparation0 = orientationCol[agentIndex, 0] * globalFrameSeparation0 + orientationCol[agentIndex, 1] * globalFrameSeparation1 
            agentFrameSeparation1 = orientationCol[agentIndex, 0] * globalFrameSeparation1 - orientationCol[agentIndex, 1] * globalFrameSeparation0 
            distanceSqr = agentFrameSeparation0 * agentFrameSeparation0 + agentFrameSeparation1 * agentFrameSeparation1
            
            # By bounding distance value we implicitly bound sensor values
            if distanceSqr < minDistanceSqr:
                distanceSqr = minDistanceSqr
            
            if distanceSqr < lowrange:
            
                # poi is east of agent
                if agentFrameSeparation0> 0:
                    # poi is north-east of agent
                    if agentFrameSeparation1 > 0:
                        observationCol[agentIndex,4] += poiValueCol[poiIndex]  / distanceSqr
                    else: # poi is south-east of agent
                        observationCol[agentIndex,7] += poiValueCol[poiIndex]  / distanceSqr
                else:  # poi is west of agent
                    # poi is north-west of agent
                    if agentFrameSeparation1 > 0:
                        observationCol[agentIndex,5] += poiValueCol[poiIndex]  / distanceSqr
                    else:  # poi is south-west of agent
                        observationCol[agentIndex,6] += poiValueCol[poiIndex]  / distanceSqr
                        
    data["Agent Observations"] = npObservationCol


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef doAgentSense_highres(data):
    cdef double PI =  3.141592655
    """
     Sensor model is <aNE, aNW, aSW, aSE, pNE, pNE, pSW, pSE>
     Where a means (other) agent, p means poi, and the rest are the quadrants
    """
    cdef int number_agents = data['Number of Agents']
    cdef int number_pois = data['Number of POIs'] 
    cdef double minDistanceSqr = data["Minimum Distance"] ** 2
    cdef double[:, :] agentPositionCol = data["Agent Positions"]
    cdef double[:] poiValueCol = data['Poi Values']
    cdef double[:, :] poiPositionCol = data["Poi Positions"]
    cdef double[:, :] orientationCol = data["Agent Orientations"]
    
    
    
    cdef int agentIndex, otherAgentIndex, poiIndex, obsIndex
    cdef double globalFrameSeparation0, globalFrameSeparation1
    cdef double agentFrameSeparation0, agentFrameSeparation1

    cdef double distanceSqr
    cdef int IDX
    cdef double theta
    cdef int RES=16

    npObservationCol = np.zeros((number_agents, RES*2), dtype = np.float64)

    cdef double[:, :] observationCol = npObservationCol
    for agentIndex in range(number_agents):

        # calculate observation values due to other agents
        for otherAgentIndex in range(number_agents):
            
            # agents do not sense self (ergo skip self comparison)
            if agentIndex == otherAgentIndex:
                continue
                theta=atan2(agentFrameSeparation0,agentFrameSeparation1)/PI
            theta = (theta+1.0)/2.0 * <float>RES
            IDX=<int>theta
            # Get global separation vector between the two agents    
            globalFrameSeparation0 = agentPositionCol[otherAgentIndex,0] - agentPositionCol[agentIndex,0]
            globalFrameSeparation1 = agentPositionCol[otherAgentIndex,1] - agentPositionCol[agentIndex,1]
            
            # Translate separation to agent frame using inverse rotation matrix
            agentFrameSeparation0 = orientationCol[agentIndex, 0] * globalFrameSeparation0 + orientationCol[agentIndex, 1] * globalFrameSeparation1 
            agentFrameSeparation1 = orientationCol[agentIndex, 0] * globalFrameSeparation1 - orientationCol[agentIndex, 1] * globalFrameSeparation0 
            distanceSqr = agentFrameSeparation0 * agentFrameSeparation0 + agentFrameSeparation1 * agentFrameSeparation1
            
            # By bounding distance value we implicitly bound sensor values
            if distanceSqr < minDistanceSqr:
                distanceSqr = minDistanceSqr
        
            
            theta=atan2(agentFrameSeparation0,agentFrameSeparation1)/PI
            theta = (theta+1.0)/2.0 * <float>RES
            IDX=<int>theta
            observationCol[agentIndex,IDX] += 1.0 / distanceSqr
                



        # calculate observation values due to pois
        for poiIndex in range(number_pois):
            
            # Get global separation vector between the two agents    
            globalFrameSeparation0 = poiPositionCol[poiIndex,0] - agentPositionCol[agentIndex,0]
            globalFrameSeparation1 = poiPositionCol[poiIndex,1] - agentPositionCol[agentIndex,1]
            
            # Translate separation to agent frame unp.sing inverse rotation matrix
            agentFrameSeparation0 = orientationCol[agentIndex, 0] * globalFrameSeparation0 + orientationCol[agentIndex, 1] * globalFrameSeparation1 
            agentFrameSeparation1 = orientationCol[agentIndex, 0] * globalFrameSeparation1 - orientationCol[agentIndex, 1] * globalFrameSeparation0 
            distanceSqr = agentFrameSeparation0 * agentFrameSeparation0 + agentFrameSeparation1 * agentFrameSeparation1
            
            # By bounding distance value we implicitly bound sensor values
            if distanceSqr < minDistanceSqr:
                distanceSqr = minDistanceSqr


            theta=atan2(agentFrameSeparation0,agentFrameSeparation1)/PI
            theta = (theta+1.0)/2.0 * <float>RES
            IDX=<int>theta
            
            observationCol[agentIndex,RES+IDX] += poiValueCol[poiIndex]  / distanceSqr
                    
    data["Agent Observations"] = npObservationCol


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.  
cpdef doAgentMove_half(data):
    cdef float worldWidth = data["World Width"]
    cdef float worldLength = data["World Length"]
    cdef int number_agents = data['Number of Agents']
    cdef double[:, :] agentPositionCol = data["Agent Positions"]
    cdef double[:, :] orientationCol = data["Agent Orientations"]
    npActionCol = np.array(data["Agent Actions"]).astype(np.float_)
    npActionCol = np.clip(npActionCol, -1, 1)*0.5
    cdef double[:, :] actionCol = npActionCol
    
    cdef int agentIndex

    cdef double globalFrameMotion0, globalFrameMotion1, norm
    
    # move all agents
    for agentIndex in range(number_agents):

        # turn action into global frame motion
        globalFrameMotion0 = orientationCol[agentIndex, 0] * actionCol[agentIndex, 0] - orientationCol[agentIndex, 1] * actionCol[agentIndex, 1] 
        globalFrameMotion1 = orientationCol[agentIndex, 0] * actionCol[agentIndex, 1] + orientationCol[agentIndex, 1] * actionCol[agentIndex, 0] 
        
      
        # globally move and reorient agent
        agentPositionCol[agentIndex, 0] += globalFrameMotion0
        agentPositionCol[agentIndex, 1] += globalFrameMotion1
        
        if globalFrameMotion0 == 0.0 and globalFrameMotion1 == 0.0:
            orientationCol[agentIndex,0] = 1.0
            orientationCol[agentIndex,1] = 0.0
        else:
            norm = sqrt(globalFrameMotion0**2 +  globalFrameMotion1 **2)
            orientationCol[agentIndex,0] = globalFrameMotion0 /norm
            orientationCol[agentIndex,1] = globalFrameMotion1 /norm
            
        # # Check if action moves agent within the world bounds
        # if agentPositionCol[agentIndex,0] > worldWidth:
        #     agentPositionCol[agentIndex,0] = worldWidth
        # elif agentPositionCol[agentIndex,0] < 0.0:
        #     agentPositionCol[agentIndex,0] = 0.0
        # 
        # if agentPositionCol[agentIndex,1] > worldLength:
        #     agentPositionCol[agentIndex,1] = worldLength
        # elif agentPositionCol[agentIndex,1] < 0.0:
        #     agentPositionCol[agentIndex,1] = 0.0
        

    data["Agent Positions"]  = agentPositionCol
    data["Agent Orientations"] = orientationCol 


 
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def assignDifferenceRewardTemporal(data):
    cdef int number_agents = data['Number of Agents']
    cdef int number_pois = data['Number of POIs'] 
    cdef double minDistanceSqr = data["Minimum Distance"] ** 2
    cdef int historyStepCount = data["Steps"] + 1
    cdef int coupling = data["Coupling"]
    cdef double observationRadiusSqr = data["Observation Radius"] ** 2
    cdef double[:, :, :] agentPositionHistory = data["Agent Position History"]
    cdef double[:] poiValueCol = data['Poi Values']
    cdef double[:, :] poiPositionCol = data["Poi Positions"]

    cdef int poiIndex, stepIndex, agentIndex, observerCount, otherAgentIndex
    cdef double separation0, separation1, closestObsDistanceSqr, distanceSqr, stepClosestObsDistanceSqr
    cdef double Inf = float("inf")
    
    cdef double globalReward = 0.0
    cdef double globalWithoutReward = 0.0
    cdef double scale = 0.0
    
    npDifferenceRewardCol = np.zeros(number_agents)
    cdef double[:] differenceRewardCol = npDifferenceRewardCol
    
    for poiIndex in range(number_pois):
        closestObsDistanceSqr = Inf
        for stepIndex in range(historyStepCount):
            # Count how many agents observe poi, update closest distance if necessary
            observerCount = 0
            stepClosestObsDistanceSqr = Inf
            for agentIndex in range(number_agents):
                # Calculate separation distance between poi and agent
                separation0 = poiPositionCol[poiIndex, 0] - agentPositionHistory[stepIndex, agentIndex, 0]
                separation1 = poiPositionCol[poiIndex, 1] - agentPositionHistory[stepIndex, agentIndex, 1]
                distanceSqr = separation0 * separation0 + separation1 * separation1
                
                # Check if agent observes poi, update closest step distance
                if distanceSqr < observationRadiusSqr:
                    observerCount += 1
                    if distanceSqr < stepClosestObsDistanceSqr:
                        stepClosestObsDistanceSqr = distanceSqr
                        
                        
            # update closest distance only if poi is observed    
            if observerCount >= coupling:
                if stepClosestObsDistanceSqr < closestObsDistanceSqr:
                    closestObsDistanceSqr = stepClosestObsDistanceSqr
        
        # add to global reward if poi is observed 
        if closestObsDistanceSqr < observationRadiusSqr:
            if closestObsDistanceSqr < minDistanceSqr:
                closestObsDistanceSqr = minDistanceSqr
            scale = (observationRadiusSqr-closestObsDistanceSqr) / (observationRadiusSqr - minDistanceSqr)
            #print("scale",scale,poiValueCol[poiIndex],poiIndex )
            globalReward += poiValueCol[poiIndex] *scale

    
    for agentIndex in range(number_agents):
        globalWithoutReward = 0
        for poiIndex in range(number_pois):
            closestObsDistanceSqr = Inf
            for stepIndex in range(historyStepCount-1,historyStepCount):
                # Count how many agents observe poi, update closest distance if necessary
                observerCount = 0
                stepClosestObsDistanceSqr = Inf
                for otherAgentIndex in range(number_agents):
                    if agentIndex != otherAgentIndex:
                        # Calculate separation distance between poi and agent\
                        separation0 = poiPositionCol[poiIndex, 0] - agentPositionHistory[stepIndex, otherAgentIndex, 0]
                        separation1 = poiPositionCol[poiIndex, 1] - agentPositionHistory[stepIndex, otherAgentIndex, 1]
                        distanceSqr = separation0 * separation0 + separation1 * separation1
                        
                        # Check if agent observes poi, update closest step distance
                        if distanceSqr < observationRadiusSqr:
                            observerCount += 1
                            if distanceSqr < stepClosestObsDistanceSqr:
                                stepClosestObsDistanceSqr = distanceSqr
                            
                            
                # update closest distance only if poi is observed    
                if observerCount >= coupling:
                    if stepClosestObsDistanceSqr < closestObsDistanceSqr:
                        closestObsDistanceSqr = stepClosestObsDistanceSqr
            
            # add to global reward if poi is observed 
            if closestObsDistanceSqr < observationRadiusSqr:
                if closestObsDistanceSqr < minDistanceSqr:
                    closestObsDistanceSqr = minDistanceSqr
                scale = (observationRadiusSqr-closestObsDistanceSqr) / (observationRadiusSqr - minDistanceSqr)
                globalWithoutReward += poiValueCol[poiIndex] * scale
        differenceRewardCol[agentIndex] = globalReward - globalWithoutReward
        
    data["Agent Rewards"] = npDifferenceRewardCol  
    data["Global Reward"] = globalReward