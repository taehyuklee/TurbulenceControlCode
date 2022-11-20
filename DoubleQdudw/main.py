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
from ou_noise import OUNoise
import critic_net
import actor_net
import os
import Environment as En

State_Episode_path = 'State & episode_unit_record/'
if not os.path.exists(State_Episode_path):
    os.makedirs(State_Episode_path)

tf.compat.v1.disable_eager_execution()

environment = En.env() #call environment
#환경을 부른다. (사실 environment라는 객체를 만든다)

_ = environment.reset(episode = 0) #환경을 초기화한다.

alpha_critic = 1.0 #learning rate (based on Q)
#alpha_actor = 0.9

#Input & Output
state_nums , state_shape = environment.state_num() #Q function은 (action, state) 이 두개에 의해 결정이 되므로, action까지 넣어줘야 한다.
action_nums, action_shape = environment.action_setting() #하나의 action 구성 list 개수를 의미한다.

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



#Reinforcement learning parmeter
n_index =5
dis = 0.99
dis_target = dis**n_index
buffer_memory = 30000 #Replay memory에 몇개를 넣을 것인가? (Buffer)
exploration_noise = OUNoise(action_shape) #ou noise도 바꿔야 하네 [action_shape = input_size_critic_a]
del_distance = 0.04
clip = 0.3

def n_step(Temporal_stack, state_step, n_index):

    global dis

    f_list = Temporal_stack[state_step-n_index]

    n_ind_list = Temporal_stack[state_step]

    reward_avg  = 0.0
    #reward_sum  = 0.0
    dis_index = 0


    #(state, action_noise, reward, next_state, end)

    for _ in range ((state_step-n_index), (state_step)):
        batch = Temporal_stack[_]
        reward_steps = batch[2]*(dis**dis_index)

        reward_avg += reward_steps/n_index
        #reward_sum += batch[2]/n_index
        dis_index +=1

    state = f_list[0]; action_noise = f_list[1]; reward =reward_avg ; next_state = n_ind_list[3]; end = n_ind_list[4]

    return state, action_noise, reward, next_state, end


def critic_train(main_critic_1, target_critic_1, main_critic_2, target_critic_2, main_actor, target_actor, train_batch):

    global dis_target

    # 학습시킬 Network와 데이터 batch가 배달옴
    Q_old = np.zeros([1], dtype = np.float64) 
    Q_new = np.zeros([1], dtype = np.float64)
    
    x_action_stack = np.empty(0)
    x_state_stack = np.empty(0)
    y_stack = np.empty(0)
    
    x_state_stack = np.reshape(x_state_stack, (0, state_shape[0]*state_shape[1]*state_shape[2]))  
    #(0, main_critic.input_size_critic_s) 였었는데 [nzp, nxp, channel] 로 뽑히니까 --> state_shape꼴로 바꾼다
    x_action_stack = np.reshape(x_action_stack, (0, action_shape[0]*action_shape[1]*action_shape[2]))
    y_stack = np.reshape(y_stack, (0, output_size_critic)) 
    #output_size_critic = 1로되어있다. target_critic.output_size_critic 굳이 class안에 있는거 뽑아 쓰기보다 그대로 사용

    for state, action, reward, next_state, end in train_batch: #이 부분들 다시 한 번 보도록 하자 (3번째 Cell에 연습함)

        scale_action = 0.2*np.std(action)

        #noise for action
        rand_num = np.random.normal(loc=0.0, scale=scale_action, size=(action_shape[0], action_shape[1], 1)) #tf.random_normal([1], mean=0, stddev=0.1)
        noisy_act = np.clip(rand_num, -clip, clip ) #tf.clip_by_value(rand_num, clip_value_min = -clip, clip_value_max = clip)
     
        #noise for action
        #rand_num = np.random.normal(loc=0.0, scale=0.2, size=(action_shape[0], action_shape[1], 1)) #tf.random_normal([1], mean=0, stddev=0.1)
        #noisy_act = np.clip(rand_num, -clip, clip ) #tf.clip_by_value(rand_num, clip_value_min = -clip, clip_value_max = clip)

        #Q_old = main_critic.predict(state[:], action[:]) #main actor

        #----------------------------------------- next_action + noise -------------------------------------------#
        #next_state_action또한 정해줘야 한다 - target policy를 이용하여 next_state에 대한걸 넣어 next_action을 유추.
        next_action = target_actor.predict(next_state[:]) #target policy로 next_state에 대한 next_action을 받아온다.
        #result of actor has shape of (n_sim, nzp*nyp*Channel)
        next_action_reshape = np.reshape(next_action, [action_shape[0], action_shape[1], action_shape[2]])
        next_act_noisy_2D = next_action_reshape[:,:,:] + noisy_act[:,:,:] #target Q을 위한 action에 noise를 줌으로써 peak overfitting을 막을수 있다.
        next_act_noisy = np.reshape(next_act_noisy_2D, [1, action_shape[0]*action_shape[1]*action_shape[2]])
        #---------------------------------------------------------------------------------------------------------#

        Q_new_1 = reward + dis_target*(target_critic_1.predict(next_state[:], next_act_noisy[:])) #target
        Q_new_2 = reward + dis_target*(target_critic_2.predict(next_state[:], next_act_noisy[:]))   
      

        Q_new = min(Q_new_1[0,0], Q_new_2[0,0])
        Q_new = np.reshape(Q_new, (1,1))

        y_stack = np.vstack([y_stack, Q_new])
        x_state_stack = np.vstack([x_state_stack, state]) #state를 학습시키는거지 Q를 학습시키는건 아니다.

        x_action_stack = np.vstack([x_action_stack, action])
        
        #actor에 input을 같이 줘야한다.
    loss_critic_1, _ = main_critic_1.update(x_state_stack, x_action_stack, y_stack) #, main_actor
    loss_critic_2, _ = main_critic_2.update(x_state_stack, x_action_stack, y_stack)
        
    return loss_critic_1, loss_critic_2, Q_old, Q_new



def actor_train(main_actor, noise_actor, main_critic, train_batch, coef_alpha, batch_size, sess):
    
    x_stack_actor = np.empty(0)
    x_stack_actor = np.reshape(x_stack_actor, (0, input_size_actor_s[0]*input_size_actor_s[1]*input_size_actor_s[2]))
    square = np.zeros([1], dtype=np.float64)
    
    for state, action, reward, next_state, end in train_batch: #이 부분들 다시 한 번 보도록 하자 (3번째 Cell에 연습함)
        
        imediate = np.zeros([1], dtype=np.float64)
        imediate = (main_actor.predict(state) - noise_actor.predict(state))**2
        square = imediate/batch_size + square

        x_stack_actor = np.vstack([x_stack_actor, state]) 

    distance = np.sqrt(np.mean(square))

    if distance < del_distance:
        coef_alpha *= 1.01
    else:
        coef_alpha /= 1.01

    _ = main_actor.update(main_critic, x_stack_actor)
        
        #서로 연결되어 있어서 main_actor update하기 위해 main_critic에 X_input를 넣어줘야한다
        #InvalidArgumentError: You must feed a value for placeholder tensor 
        #'main_critic/input_critic_state' with dtype float and shape [?,180]
        
    return coef_alpha, distance   


def first_copy (sess, target_scope_name ="target", main_scope_name = "main"):

    op_holder = []
    
    main_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=main_scope_name)
    target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=target_scope_name)

    for main_var, target_var in zip(main_vars, target_vars): 
        
        op_holder.append(target_var.assign(main_var.value()))

    return sess.run(op_holder)



def copy_var_ops(*, target_scope_name ="target", main_scope_name = "main"):

    op_holder = []
    tau = 0.001
    
    main_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=main_scope_name)
    target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=target_scope_name)
    '''
    print("main_vars")
    print(main_vars)
    print("target_vars")
    print(target_vars)
    '''

    for main_var, target_var in zip(main_vars, target_vars): 
        
        op_holder.append(target_var.assign(tau * main_var.value() +(1 - tau)*target_var.value()))

            
        
    return op_holder, main_var.value(), target_var.value()



def space_noise(noise_vars, noise_name ="noise_actor", main_name = "main_actor"):

    noise_stack = []
    noise_added_stack = []

    main_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=main_name)
    noise_added_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=noise_name)

    for main_var, noise_var in zip(main_vars, noise_vars): 
               
        noise_stack.append(noise_var.assign(noise_var + 0.5* (0 - noise_var) + tf.random_normal(tf.shape(main_var), mean = 0.0, stddev = (alpha)*tf.math.reduce_std(main_var),dtype=tf.float32)))
        #episode_decay #step_decay

    for main_var, noise_var, noise_added_var in zip(main_vars, noise_vars, noise_added_vars): 
        noise_added_stack.append(noise_added_var.assign(main_var + noise_var))
  
    return noise_stack, noise_added_stack


def get_noise_var():  
    vars1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "main_actor")
    noise_vars = [tf.Variable(tf.zeros(var.shape,dtype=tf.float32),dtype=tf.float32) for var in vars1]
    return noise_vars 

def main():
    
    global n_index

    Q_old = np.empty(0)
    Q_new = np.empty(0)  
    Temporal_stack = []
    state_step = 0
    Loss_step = 0
    main_update_freq = 1
    actor_update_freq = 5
    target_update_frequency = 5
    train_loop_epoch = 1
    #main이 target을 향해서 update되어가고 이후에 target_update가 이루어져야 하기때문에 main_freq < target_update가 되어야 한다.
    max_episodes = 25
    batch_size = 64 #Mini batch size Buffer에서 몇개씩 batch로 만들어서 학습시킬 것인가?
    buff_len = batch_size    
    temp_1=0 #recording system
    total_time = 0
    coef_alpha = 0.1
    append_t = False



    #------------ state time interval ------------#
    st_step = 50
    #action을 몇 time-step마다 취할 것인지에 대한 숫자 <one state_step = delt (st_step)>
    step_deadline = 1200 #state_step의 deadline
    ##참고로 time_step에 대한 deadline은 st_step *step_deadline으로 구하면 된다 <real_time step이 중요함>
    starting_act = batch_size*st_step
    #나중에 학습이 다 끝나고 test해 볼때의 test time step
    noise_actor_freq = 5
    #---------------------------------------------#
    
    #------------ recording parameter ----------#
    video_record_freq = 1 #50 Episode마다 Video로 촬영해서 저장한다.
    snap_shot_freq = 2000 #(st_step*step_deadline) #0부터 시작이기때문에 뒤에 1을 더 해준거, 매 Episode마다 10000 times에 한 번 촬영
    #-------------------------------------------#

    # Replay buffer를 deque로 짠다. 
    buffer = deque(maxlen =buffer_memory) 
    #Memory는 50000개까지 

    #reward_buffer = deque() #maxlen=100
    #reward_buffer또한 deque로 만들어서 마지막 100개까지 기억하도록 한다

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
        
        noise_vars = get_noise_var() #중간 매개역할 하는 nosie weight variables를 선언한 것
        
        _ = main_actor.initialization_a(main_critic_1.Objective, name ="main_actor")
        _ = target_actor.initialization_a(target_critic_1.Objective, name ="target_actor")
        _ = main_critic_1.initialization_c(name ="main_critic_1")
        _ = main_critic_2.initialization_c(name ="main_critic_2")
        _ = target_critic_1.initialization_c(name ="target_critic_1")
        _ = target_critic_2.initialization_c(name ="target_critic_2")     
        
        #actor_var_save = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="main_actor")
        saver_act = tf.train.Saver(max_to_keep=None) #save model (객체)
        #var_list=actor_var_save, 

        sess.run(tf.global_variables_initializer()) #initializer <여기서 전체적으로 초기화해준다.>
        print("initialization complete")
        
        #Critic (first_copy)
        _ = first_copy(sess, target_scope_name="target_critic_1",main_scope_name="main_critic_1")
        _ = first_copy(sess, target_scope_name="target_critic_2",main_scope_name="main_critic_2")

        #Policy (first_copy)
        _ = first_copy(sess, target_scope_name="target_actor", main_scope_name="main_actor")
        
        #Critic (Copy)
        copy_critic_1, main_val_c, target_val_c = copy_var_ops(target_scope_name="target_critic_1",main_scope_name="main_critic_1")
        copy_critic_2, main_val_c, target_val_c = copy_var_ops(target_scope_name="target_critic_2",main_scope_name="main_critic_2")
        
        #Policy (Copy)
        copy_actor, main_val_a, target_val_a =  copy_var_ops(target_scope_name="target_actor",main_scope_name="main_actor")

        #Noise 
        noise_copy, noise_added_copy = space_noise(noise_vars, noise_name ="noise_actor", main_name = "main_actor")


        for episode in range(0, max_episodes+1):
            
            print("Episode : {} start ".format(episode))    
            # 처음에 다시 형태 잡아주고 end를 False로 시작해준다
            time = 0 #time check for update
            end  = False
            state = np.zeros([state_shape[0], state_shape[1], state_shape[2]], dtype = np.float64)
            next_state = np.zeros([state_shape[0], state_shape[1], state_shape[2]], dtype = np.float64)
            action = np.zeros([action_shape[0], action_shape[1], action_shape[2]], dtype = np.float64)


            ##################### environment로부터 state를 받아온다 (observation) ###############
            state_stock = environment.reset(episode) #envrionment로부터 state를 가져온다. (초기 state)
            exploration_noise.reset()
            
            #state_b_du = state[0]
            #state_b_dw = state[2]
            #state_t_du = state[1]
            #state_t_dw = state[3]

            #Initial state를 [nzp, nxp] 이미지로 준비하는 부분
            state[:,:,0] = state_stock[0]; state[:,:,1] = state_stock[2] #state에 대한 RGB처럼 두개의 구성셋으로 구성함
            #print("state outside")
            #print(state)
            state = np.reshape(state, [1, state_shape[0]*state_shape[1]*state_shape[2]]) #일단 그냥 한줄로 변형해서 처리해주는게 편하다
            #state_shape
            #action_shape

            reward_graph = 0

            reward_record = open(State_Episode_path + "/reward.plt" , 'a', encoding='utf-8', newline='') 
            if temp_1 ==0: reward_record.write('VARIABLES = "Episode", "Reward" \n') 

            #Episode bulk graph    
            Bulk_record = open(State_Episode_path + "/Bulk_velocity (Ep unit).plt" , 'a', encoding='utf-8', newline='')
            if temp_1==0: Bulk_record.write('VARIABLES = "Episode", "Bulk_vel" \n') 


            temp_1=1 #variable을 한 번만 저장하기 위함
            #Reward를 기록하기 위함.                               
            #Noise 그래프 그리기        

            noise_record = open(State_Episode_path + "/noise, episode{}.plt" .format(episode), 'a', encoding='utf-8', newline='')
            noise_record.write('VARIABLES = "state_step", "noise" \n')  

            #이 안에서 state, next_state형태는 [1, nzp*nxp*channels] 이 형태로 계속 돌아다닌다.
            while not end == True: #이부분이 한 episode에서 state를 진행시키는 부분 actor critic을 이 부분에 넣어야함.
                
                #State reward graph
                #state_reward_record = open(State_Episode_path + "/state_reward, episode{}.plt" .format(episode), 'a', encoding='utf-8', newline='')
                state_reward_record = open(State_Episode_path + "/state_reward.plt", 'a', encoding='utf-8', newline='')
                #Loss for critic 
                Loss_record = open(State_Episode_path + "/Loss_record.plt", 'a', encoding='utf-8', newline='')
                alpha_value = open(State_Episode_path + "/alpha_value (state unit).plt" , 'a', encoding='utf-8', newline='')                

                if episode == 0 and state_step == 0:
                    Loss_record.write('VARIABLES = "state", "Loss" \n')  

                if state_step == 0: 
                    state_reward_record.write('VARIABLES = "state_step", "avg_reward" \n') 
                    alpha_value.write('VARIABLES = "state_step", "coef_alpha" "distance"\n') 
                    

                #Noise reset하기
                if state_step % noise_actor_freq == 0:
                    state_step_f =  np.reshape(state_step, (1)) #state_step_f =  np.reshape(total_time, (1)) #
                    coef_f = np.reshape(coef_alpha, (1))
                    episode_f = np.reshape(episode, (1))
                    feed = {step_decay: state_step_f, episode_decay: episode_f, alpha: coef_f} 
                    sess.run([noise_copy], feed_dict=feed)
                    sess.run([noise_added_copy], feed_dict=feed)
                    total_time = total_time + 1 #it will be used next state step

                #next_state에 대해서 해줘야한다 why? next_state는 계속해서 한 Episode내에서 environment로부터 받아오기때문에 받아오는 형태를 그대로 받와야하기때문에 
                next_state = np.zeros([state_shape[0], state_shape[1], state_shape[2]], dtype = np.float64)
                
                ############### 두개의 Neural network로 학습을 시키는 부분이다 ##########
                if append_t == True:
                    if len(buffer) > buff_len and state_step % main_update_freq == 0: # train every 10 episodes
                        #print("update start")
                        loss_avg = 0
                    
                        for _ in range(train_loop_epoch):
                            #print("random_sample, step :{}" ,format(_)) #check complete
                            minibatch = random.sample(buffer, batch_size) 
                            minibatch = list(minibatch)
                        
                            #print("critic update start")
                            loss_critic, _ , Q_old, Q_new= critic_train(main_critic_1, target_critic_1, main_critic_2, target_critic_2, main_actor, target_actor, minibatch)
                        
                            #print("actor update start")
                            if state_step % actor_update_freq ==0:
                                coef_alpha, distance  = actor_train(main_actor, noise_actor, main_critic_1, minibatch, coef_alpha, batch_size, sess)
                                alpha_value.write("%d %f %f \n" %(state_step, coef_alpha, distance))
                                alpha_value.close()

                            loss_avg = loss_critic/train_loop_epoch +loss_avg

                        print("Loss for critic is : {}".format(loss_avg))   
                        Loss_record.write("%d %f \n" %(Loss_step , loss_avg))
                        Loss_record.close()
                        #print("update end")

                                    #main을 target으로 복사한다 (critic)
                    if state_step > 1 and state_step % target_update_frequency == 0:
                        sess.run(copy_critic_1)
                        sess.run(copy_critic_2)
                    
                    #main을 target으로 복사한다 (actor)
                    if state_step > 1 and state_step % target_update_frequency == 0:
                        sess.run(copy_actor)
                        if episode == 0 and state_step < buff_len:
                            pass
                        else:
                            print("target update")
                        #if state_step > buff_len:
                ########################################################################
                
                
                
                #Noise 매 step마다 Noise의 정도는 작아지게 설정할 것이다.
                Noise = 0*exploration_noise.noise() # 매 step마다 Normal distribution에서 임의로 추출한다.
                Noise = Noise/((state_step*0.01+ episode*1+1))
                
                ##################### deterministic policy에서 state를 주고 action을 뽑아온다 ###################
                action = noise_actor.predict(state) + Noise

                #state [1, nzp*nxp*nums of state]
                #action [1, nzp*nxp*nums of action]

                #critic, action에 넣어줄때는 일단 모두 [1, N]의 형태로 넣어줘야 하므로 다음과 같이 
                action_noise = np.reshape(action, (action_shape[0]*action_shape[1]*action_shape[2])) #이건 buffer에 넣어줄 것이다. 

                #noise를 입력한다
                noise_record.write("%d %f \n" %(state_step ,np.mean(Noise)))


                
                     
                #여기는 actor가 잘 돌아가는지 확인하는 부분. tensorflow는 trainable 초기화는 항상 해주고 
                #placeholder에 X_input등을 넣어주면된다 (feed) 근데 여기서는 state가 input이니까 나오는게 맞음 feed안해도
                
                # Get new state and reward from environment  
                next_state_stock, reward, bulk_v_return, end = environment.simulation(state, state_step, noise_actor, Noise, st_step, episode, video_record_freq, snap_shot_freq, step_deadline, starting_act)
                
                next_state[:,:,0] = next_state_stock[0]; next_state[:,:,1] = next_state_stock[2] #next_state에 대한 RGB처럼 두개의 Channel  

                #next_state에 대해 [1, nzp*nxp*nums]로 바꿔줘야한다
                next_state = np.reshape(next_state, [1, state_shape[0]*state_shape[1]*state_shape[2]]) #일단 그냥 한줄로 변형해서 처리해주는게 편하다

               
                # break part
                if end ==True:
                    break


                ################ 이 부분이 N-time Replay memory 부분이다 ##############

                state = np.reshape(state, [1, state_shape[0]*state_shape[1]*state_shape[2]])

                stack_element = (state, action_noise, reward, next_state, end)
                Temporal_stack.append(stack_element) #Stack 


                #state, action 모두 [1, nzp*nxp*nums_of_set] 이렇게 넘겨준다
                ################ 이 부분이 Replay memory 부분이다 ##############

                if state_step >= (n_index):

                    state, action_noise, reward_p, next_state_p, end = n_step(Temporal_stack, state_step, n_index)
                    buffer.append((state, action_noise, reward_p, next_state_p, end))

                    append_t = True

                    #한 step의 reward씩 계속 reward_graph에 쌓는다. summation of reward
                    reward_graph = reward_p + reward_graph

                    #한 Episode에서 순간 reward를 기록하기 위함
                    state_reward_record.write("%d %f \n" %(state_step ,reward_p))
                    state_reward_record.close() #어차피 append니까 바로 닫아도 됨

                    if len(buffer) > buffer_memory:
                        buffer.popleft()

                ################################################################       
                #print(sess.run(main_val_c))
                #print(sess.run(target_val_a))
               
                state = next_state

                if state_step == step_deadline-1:
                    break #<Simulator안에서 step_deadline을 넣어서 끝낼 필요는 없다>

                state_step = state_step + 1
                time = time + 1
                Loss_step += 1
                #print("step num : {}".format(step))        
           
            #model save each episode
            saver_act.save(sess, './Save_check/model episode {}'.format(episode), global_step = None)
            
            #한 Episode가 끝나고 정리를 하는 곳
            bulk_v_graph = np.mean(bulk_v_return)
            reward_graph = reward_graph/state_step
            
            Temporal_stack = [] #Initialization of Temporal stack

            #plt file로 reward graph 저장
            reward_record.write("%d %f \n" %(episode , reward_graph))
            Bulk_record.write("%d %f \n" %(episode , bulk_v_graph))
            
            #기록을 닫는것 
            noise_record.close()
            Bulk_record.close()
            
            state_step = 0
            #print("Episode : {} end ".format(episode))
            
            reward_record.close()  


        #Let's go to test_set with learned actor
        #---------------------------------------------------------- Test part ---------------------------------------------------------#
                
        #--------------------------- Test parameter ---------------------#
        end_step = 350000
        episode_test = "Test"
        #Video & Snap shot parameter
        video_record_freq = 1 #50 Episode마다 Video로 촬영해서 저장한다.
        snap_shot_freq = 1000 #(st_step*step_deadline) #0부터 시작이기때문에 뒤에 1을 더 해준거, 매 Episode마다 10000 times에 한 번 촬영
        #----------------------------------------------------------------#

        _ = environment.reset(episode="Test") #reset environment

        '''
        state = np.zeros([state_shape[0], state_shape[1], state_shape[2]], dtype = np.float64)
        state_stock = environment.reset() #reset environment
        state[:,:,0] = state_stock[0]; state[:,:,1] = state_stock[2] #state 
        state = np.reshape(state, [1, state_shape[0]*state_shape[1]*state_shape[2]]) #초기 state reshape
        '''
        #Test_performance of Actor
        environment.test_performance(main_actor, end_step, episode_test, video_record_freq, snap_shot_freq)

        print("Test is finished")
        #------------------------------------------------------------------------------------------------------------------------------#
            

if __name__ == "__main__":

    main()

    print("All process is finished!")