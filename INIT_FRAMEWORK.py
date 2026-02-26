from smac.env import StarCraft2Env

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from collections import deque,namedtuple

class SharedReplayBuffer:
    """
    1. Reward
    2. Global State
    3. Terminated 
    """
    
    def __init__(self,warmup_length = 100000): 
        self.trans  = namedtuple("Global_Information",("step","global_state","joint_action","reward","terminated"))  
        self.buffer = deque([],maxlen = warmup_length)
        
    def push(self,*args):
        self.buffer.append(self.trans(*args))
        
    def __len__(self):
        return len(self.buffer) 
          
class ReplayBuffer:
    def __init__(self,warmup_length = 100000):
        self.trans = namedtuple("Local_Information",("agent_id","step","obs","action"))
        self.buffer = deque([],maxlen=warmup_length)
    
    def push(self,*args):
        self.buffer.append(self.trans(*args))
        
    def __len__(self):
        return len(self.buffer)
        
class LocalRepresentationModel(nn.Module):
       
    def __init__(self,in_feat):
        super().__init__()
        out_feat = in_feat // 2
        self.fc1 = nn.Linear(in_feat, out_feat)
        
        in_feat = out_feat 
        out_feat = in_feat // 2
        self.fc2 = nn.Linear(in_feat,out_feat)
        
        in_feat = out_feat
        out_feat = in_feat // 2
        self.fc3 = nn.Linear(in_feat, out_feat)
        
        self.act = nn.LeakyReLU(0.1)

        self.mean = nn.Linear(out_feat,out_feat)
        self.sigma   = nn.Linear(out_feat,out_feat)
        
        in_feat = out_feat
        out_feat = in_feat * 2
        self.dfc3 = nn.Linear(in_feat,out_feat)
        
        in_feat = out_feat 
        out_feat = in_feat * 2
        self.dfc2 = nn.Linear(in_feat, out_feat)
        
        in_feat = out_feat
        out_feat = in_feat * 2
        self.dfc1 = nn.Linear(in_feat, out_feat)
        
        
    def encoder(self,x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        
        return self.mean(x), self.sigma(x)
    
    def sampling(self,x):
        mean, sigma = self.encoder(x)
        eps = torch.randn_like(sigma)
        return mean + eps * sigma
    
    def decoder1(self,x):
        x = self.act(self.dfc3(x))
        x = self.act(self.dfc2(x))
        x = self.act(self.dfc1(x))
        return x
        
    def decoder2(self,x):
        # Good Question
        pass
    
    def forward(self,x):
        shared_latent_z = self.sampling(x)
        reconstruct_obs = self.decoder1(shared_latent_z)
        # global_latent_z = self.decoder2(shared_latent_z)
        # return reconstruct_obs, global_latent_z
        return reconstruct_obs
        

class GlobalRepresentationModel:
    def __init__(self):
        pass
    
    def forward():
        pass

class LocalDynamicsModel:
    pass

class GlobalDynamicsModel:
    pass



    
    

def main():
    
    
    print("Setup all the things.")
    env = StarCraft2Env(map_name = "8m") # environment, map_name take task name
    env_info = env.get_env_info()
    
    n_agents = env_info["n_agents"]
    
    env.reset()
    local_in_feat = env.get_obs()[0]
    global_in_feat =  env.get_state()
    
    print(local_in_feat.shape[0])
    training_epoch = 100
    
    warmup_length = 1000
    step = 0
    collect_data_bar = tqdm(total = warmup_length)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    
    local_info_memory = ReplayBuffer(n_agents * warmup_length)
    global_info_memory = SharedReplayBuffer(warmup_length)
    
    local_RM = LocalRepresentationModel(in_feat = 80).to(device)
    global_RM = GlobalRepresentationModel()
    local_DM = LocalDynamicsModel()
    global_RM = GlobalDynamicsModel()
    
    print("Start to Collect ")
    
    while step < warmup_length:

        env.reset()
        terminated = False
        
        while not terminated:
            obs = env.get_obs()
            state = env.get_state()
            
            actions = []
            
            for agent_id in range(n_agents):
                avail_actions = env.get_avail_agent_actions(agent_id)
                avail_actions_ind = np.nonzero(avail_actions)[0]
                action = np.random.choice(avail_actions_ind)
                actions.append(action)
                
                agent_obs = obs[agent_id]
                local_info_memory.push(agent_id, step, agent_obs, action)
                
            reward, terminated, _ = env.step(actions)
            
            
            global_info_memory.push(step,state,actions,reward,terminated)
            step += 1
            collect_data_bar.update(1)
        
    ### training
    # for ep in range(training_epoch):
    obs_np = np.array([rb.obs for rb in local_info_memory.buffer])
    agent_id_np = np.array([rb.agent_id for rb in local_info_memory.buffer])
    reward_np = np.array([shared_rb.reward for shared_rb in global_info_memory.buffer])
    joint_action_np = np.array([shared_rb.joint_action for shared_rb in global_info_memory.buffer])
    terminated_np =  np.array([shared_rb.terminated for shared_rb in global_info_memory.buffer])
    stat_np =  np.array([shared_rb.global_state for shared_rb in global_info_memory.buffer])
    
    
    obs_buffer = torch.from_numpy(obs_np).to(device)    
    agent_id_buffer = torch.from_numpy(agent_id_np).to(device) 
    reward_buffer = torch.from_numpy(reward_np).to(device) 
    joint_action_buffer = torch.from_numpy(joint_action_np).to(device) 
    terminated_buffer = torch.from_numpy(terminated_np).to(device) 
    stat_buffer = torch.from_numpy(stat_np).to(device)
    
    print()
    print(f"Local Observation shape: {obs_buffer.shape}")
    print(f"Agent Id Shape: {agent_id_buffer.shape}")
    print(f"Reward Shape: {reward_buffer.shape}") 
    print(f"Joint Action Shape: {joint_action_buffer.shape}")
    print(f"Terminared Shape: {terminated_buffer.shape}") 
    print(f"Global Status Shape: {stat_buffer.shape}")
    
    print("Start to training")
    # for ep in tqdm(range(training_epoch)):
    reconstruct_obs = local_RM(obs_buffer)
    
    print(reconstruct_obs.shape)    
    
    env.close()

if __name__ == "__main__":
    main()
    