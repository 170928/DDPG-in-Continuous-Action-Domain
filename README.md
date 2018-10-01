# DDPG-in-Continuous-Action-Domain  
[Implementation]  
  
State_dim : [......] 형태의 Continous State Domain  
Action_dim : [a, b] 형태의 Continuos Action Domain  

Actor의 Policy 를 통해서 1x2 tensor의 action (output of policy)를 선택하게 합니다.  
action (output)은 sigmoid activation을 거쳐 나온 1x2 tensor 가 되며,  
원하는 action boundary값을 설정하여   
action_boundary * action 과 같은 방법으로 regression 하여 [a, b]의 output 값을 boundary에 맞게 변형합니다.  
