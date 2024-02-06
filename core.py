"""
SimulationCore class is the backbone of any arbitrary domain.
By default, SimulationCore does not provide any domain-specific functionality.
To provide functionality, modify the following attributes:

data:
A dictionary shared amongst all functions in the simulation.
User may add any property they may want to have shared by all provided functions
SimulationCore provides and manages the following keys during run() execution:
    "Steps": duration of world measured in time steps,
    "Trains per Episode":  number of world instances for training to generate
        in sequence each episode
    "Tests per Episode":  number of world instances for testing to generate
        in sequence each episode
    "Number of Episodes": number of episodes (i.e. generations) in the trial
    "Episode Index": the index of the current episode in the trial
    "Mode": the current simulation mode which can be set to "Train" or "Test"
        Training mode runs before testing mode
    "World Index": the index of the current world instance in the current mode
        and episode
    "Step Index": the index of the current time step for the current world 
        instance
Warning: Use caution when manually reseting these values within the simulation.


Note: Each function (or callable class) must take in the dictionary data as its 
    first and only required parameter. 

trialBeginFuncCol:
An ordered collection of functions; each function is executed in order at the 
    beginning of the trial. These functions should set the entire simulation
    trial.

trainBeginFuncCol:
An ordered collection of functions; each function is executed in order at the 
    beginning of the training mode of the current episode. These functions 
    should set the episode for training. Train mode runs before test mode.
    
worldTrainBeginFuncCol:
An ordered collection of functions; each function is executed in order at the 
    beginning of the world instance of the current episode when in training 
    mode. These functions should set the world.

worldTrainStepFuncCol:
An ordered collection of functions; each function is executed in order  
    for each step in the current world instance when in training 
    mode.
    
self.worldTrainEndFuncCol:
An ordered collection of functions; each function is executed in order  
    at the end of the world instance of the current episode when
    in training mode. 

self.trainEndFuncCol:
An ordered collection of functions; each function is executed in order  
    at the end of the training mode of the current episode. 
    
testBeginFuncCol:
An ordered collection of functions; each function is executed in order at the 
    beginning of the testing mode of the current episode. These functions 
    should set the episode for testing. Train mode runs before test mode.
    
worldTrainBeginFuncCol:
An ordered collection of functions; each function is executed in order at the 
    beginning of the world instance of the current episode when in testing 
    mode. These functions should set the world.

worldTrainStepFuncCol:
An ordered collection of functions; each function is executed in order  
    for each step in the current world instance when in testing 
    mode.
    
self.worldTrainEndFuncCol:
An ordered collection of functions; each function is executed in order  
    at the end of the world instance of the current episode when
    in testing mode. 

self.testEndFuncCol:
An ordered collection of functions; each function is executed in order  
    at the end of the testing mode of the current episode. 

trialEndFuncCol:
An ordered collection of functions; each function is executed in order at the 
    end of the trial. Some of these functions should save important trial
    information.

Note: Each function must take in the dictionary data as its first and only required 
    parameter. 
"""
class SimulationCore:
    def __init__(self):
        """
        Run function executes a new simulation trial by runnning prescribed 
            functions at prescribed times
        
        Args:
           
        Returns:
            None
        """
        self.data = {
            "Steps": 5,
            "Trains per Episode": 3,
            "Tests per Episode": 1,
            "Number of Episodes": 20
        } 
    
        self.trialBeginFuncCol = []
        
        self.trainBeginFuncCol = []
        self.worldTrainBeginFuncCol = []
        self.worldTrainStepFuncCol = []
        self.worldTrainEndFuncCol = []
        self.trainEndFuncCol = []
        
        self.testBeginFuncCol = []
        self.worldTestBeginFuncCol = []
        self.worldTestStepFuncCol = []
        self.worldTestEndFuncCol = []
        self.testEndFuncCol = []
        
        self.trialEndFuncCol = []
        
    def run(self):
        """
        Run function executes the simulation but runnning prescribed functions 
            at prescribed times
        
        Args:
           
        Returns:
            None
        """
        
        # Do Trial Begin Functions
        for func in self.trialBeginFuncCol:
            func(self.data)
            
        # Do Each Episode
        for episodeIndex in range(self.data["Number of Episodes"]):
            self.data["Episode Index"] = episodeIndex
            
            # Do Begin Training Functions
            for func in self.trainBeginFuncCol:
                func(self.data)
    
            # Repeat running world (with new teams) until repeat is set to false
            for worldIndex in range(self.data["Trains per Episode"]):
                self.data["Mode"] = "Train"
                self.data["World Index"] = worldIndex
                self.data["Step Index"] = None
                
                # Do world begin (setup) functions
                for func in self.worldTrainBeginFuncCol:
                    func(self.data)
                
                # Do world end functions
                for stepIndex in range(self.data["Steps"]):
                    self.data["Step Index"] = stepIndex
                    for func in self.worldTrainStepFuncCol:
                        func(self.data)
                    
                # Do world 
                for func in self.worldTrainEndFuncCol:
                    func(self.data)
    
            # Do End Training Functions
            for func in self.trainEndFuncCol:
                func(self.data)
            
            # Do Begin Testing Functions
            for func in self.testBeginFuncCol:
                func(self.data)
    
            # Repeat running world (with new teams) until repeat is set to false
            for worldIndex in range(self.data["Tests per Episode"]):
                self.data["Mode"] = "Test"
                self.data["World Index"] = worldIndex
                self.data["Step Index"] = None
                
                
                # Do world begin (setup) functions
                for func in self.worldTestBeginFuncCol:
                    func(self.data)
                
                # Do world end functions
                for stepIndex in range(self.data["Steps"]):
                    self.data["Step Index"] = stepIndex
                    for func in self.worldTestStepFuncCol:
                        func(self.data)
                    
                # Do world 
                for func in self.worldTestEndFuncCol:
                    func(self.data)
    
            # Do End Testing Functions
            for func in self.testEndFuncCol:
                func(self.data)
                
        # Do Trial End Functions
        for func in self.trialEndFuncCol:
            func(self.data)
            


