#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import tensorflow.compat.v1 as tf
import random
from collections import deque
import critic_net
import actor_net
import os
import Environment as En

# The file path to save the data
save_file = './model episode 8.ckpt'

tf.compat.v1.disable_eager_execution()

environment = En.env() #call environment
#환경을 부른다. (사실 environment라는 객체를 만든다)

#Input & Output
state_nums , state_shape = environment.state_num() #Q function은 (action, state) 이 두개에 의해 결정이 되므로, action까지 넣어줘야 한다.
action_nums, action_shape = environment.action_setting() #하나의 action 구성 list 개수를 의미한다.


_ = environment.reset(episode = "Test") #환경을 초기화한다.

alpha_critic = 1.0 #learning rate (based on Q)


input_size_critic_s = state_shape #[nzp, nxp, numbers of states set]
input_size_critic_a = action_shape #[nzp, nxp, numbers of action set]
output_size_critic = 1 #DQN과 다르게 모든 action에 대한 Q를 계산하는 것이 아니라, Policy가 최적의 action을 선택해주기때문에 
                       #하나의 Q값이 생긴다 (여느 state에 대한 최적의 action에 대해서)
input_size_actor_s = state_shape #[nzp, nxp, numbers of states set]
output_size_actor =  action_shape #[nzp, nxp, numbers of action set]

#각 State, action shape을 받와와서 그대로 만들어준다.
state = np.zeros([state_shape[0], state_shape[1], state_shape[2]], dtype = np.float64)
next_state = np.zeros([state_shape[0], state_shape[1], state_shape[2]], dtype = np.float64)
action = np.zeros([action_shape[0], action_shape[1], action_shape[2]], dtype = np.float64)
step_decay = tf.placeholder(tf.float32, [1])
episode_decay = tf.placeholder(tf.float32, [1])
alpha = tf.placeholder(tf.float32, [1])


def main():

    #------------ recording parameter ----------#
    video_record_freq = 1 #50 Episode마다 Video로 촬영해서 저장한다.
    snap_shot_freq = 1000 #(st_step*step_deadline) #0부터 시작이기때문에 뒤에 1을 더 해준거, 매 Episode마다 10000 times에 한 번 촬영
    #-------------------------------------------#

    config = tf.ConfigProto(log_device_placement=True)	
    config.gpu_options.allow_growth = True


    with tf.Session(config=config) as sess:


        #formation of network for actor net
        main_actor = actor_net.actor(sess, input_size_actor_s, output_size_actor, output_size_critic, name="main_actor") 
        target_actor = actor_net.actor(sess, input_size_actor_s, output_size_actor, output_size_critic, name="target_actor")  
        noise_actor = actor_net.actor(sess, input_size_actor_s, output_size_actor, output_size_critic, name="noise_actor")  
       
        #formation of network for critic net (first error NameError - input_size ciritic 등)
        main_critic_1 = critic_net.critic(sess, input_size_critic_s,input_size_critic_a, output_size_critic, main_actor, name="main_critic_1") 
        main_critic_2 = critic_net.critic(sess, input_size_critic_s,input_size_critic_a, output_size_critic, main_actor, name="main_critic_2") 
        target_critic_1 = critic_net.critic(sess, input_size_critic_s,input_size_critic_a, output_size_critic, target_actor, name="target_critic_1")    
        target_critic_2 = critic_net.critic(sess, input_size_critic_s,input_size_critic_a, output_size_critic, target_actor, name="target_critic_2")   
        #main_actor.action_pred를 줌으로써 이어줘 본다.

        _ = main_actor.initialization_a(main_critic_1.Objective, name ="main_actor")
        _ = target_actor.initialization_a(target_critic_1.Objective, name ="target_actor")
        _ = main_critic_1.initialization_c(name ="main_critic_1")
        _ = main_critic_2.initialization_c(name ="main_critic_2")
        _ = target_critic_1.initialization_c(name ="target_critic_1")
        _ = target_critic_2.initialization_c(name ="target_critic_2")    

        saver_act = tf.train.Saver(max_to_keep=None) #save model (객체)
        #var_list=actor_var_save, 

        sess.run(tf.global_variables_initializer()) #initializer <여기서 전체적으로 초기화해준다.>
        print("initialization complete")

        # Remove the previous weights and bias
        #tf.reset_default_graph()

        # Class used to save and/or restore Tensor Variables
        saver = tf.train.Saver()

        path = tf.train.latest_checkpoint("save_check")
        print(path)

        # Load the weights and bias
        ckpt_path = saver.restore(sess, path)
        print("loaded")

        #Let's go to test_set with learned actor
        #---------------------------------------------------------- Test part ---------------------------------------------------------#
        #어차피 test part는 reset하고 actor만 넘겨주면 알아서 다 되게 해놨다 - 위에서 Parameter load해와서 하면 됨.

        #--------------------------- Test parameter ---------------------#
        end_step = 300001
        episode_test = "Test"
        #Video & Snap shot parameter
        video_record_freq = 1 #50 Episode마다 Video로 촬영해서 저장한다.
        snap_shot_freq = 1000 #(st_step*step_deadline) #0부터 시작이기때문에 뒤에 1을 더 해준거, 매 Episode마다 10000 times에 한 번 촬영
        time_average_freq = 5000
        scatter_freq = 1000
        #----------------------------------------------------------------#

        _ = environment.reset(episode="Test") #reset environment

        #Test_performance of Actor
        environment.test_performance(main_actor, end_step, episode_test, video_record_freq, snap_shot_freq, scatter_freq)

        print("Test is finished")
        #------------------------------------------------------------------------------------------------------------------------------#
            

if __name__ == "__main__":

    main()

    print("All process is finished!")