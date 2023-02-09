#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow as tf_2


class actor:

    def __init__(self, session, input_size_actor_s, output_size_actor, output_size_critic,  name="main"): 

        #객체 변수들에 대해서 다음과 같이 정의 한다. session, input_size, output_size, name 
        self.session = session
        self.input_size_actor_s = input_size_actor_s #[nzp, nxp, numbers of states set]
        self.output_size_actor = output_size_actor #[nzp, nxp, numbers of action set]
        #self.output_size_critic = output_size_critic
        self.net_name = name
        
        print(name)
        self.build_network() # class를 만들면 알아서 객체의 Neural network를 만들게 해 놓은거임.
        


    def build_network(self, k_1size = 64, k_2size=32, k_3size=1, k_4size=1): 
#k_2size가 한개 이미지 하나 

        def periodic_pad(X, padding_size=1): 
            X = tf.concat([X[:,-padding_size:,:,:], X, X[:,0:padding_size,:,:]], axis=1)
            X = tf.concat([X[:,:,-padding_size:,:], X, X[:,:,0:padding_size,:]], axis=2)
            return X
    
        with tf.variable_scope(self.net_name, reuse = tf.AUTO_REUSE): 
            #변수의 범위는 변수가 효력을 미치는 영역을 의미한다 (변수의  scope) -namespace벗어나면 영향력이 없어진다.
            
            #[nzp, nxp, 3]
            self.X_input = tf.placeholder(tf.float32, [None, self.input_size_actor_s[0]*self.input_size_actor_s[1]*self.input_size_actor_s[2]], name="X_input")
            self.X_input_2 = tf.reshape(self.X_input, [-1, self.input_size_actor_s[0], self.input_size_actor_s[1], self.input_size_actor_s[2]]) 
            #지금 아마 이것때문에 문제가 생긴것 같다. 
            
            
            # Configuration of Conv layer 1
            
            k1_S = [3, 3, self.input_size_actor_s[2], k_1size]
            k1_std = np.sqrt(2)/np.sqrt(np.prod(k1_S[:-1]))
            
            #filter [filter_height, filter_width, in_channels, out_channels] 
            self.kernal_1 = tf.get_variable("W_a1", shape =[3 , 3, self.input_size_actor_s[2], k_1size], initializer = tf.random_normal_initializer(mean=0.0, stddev=k1_std, seed=None))
            self.B_a1 = tf.get_variable("B_a1", shape = [1,1, 1, k_1size], initializer = tf.initializers.zeros())
            #1D로 Kernal을 만들어야 하니까 [1,3]으로 한다.
            
            linear_1 = tf.nn.conv2d(periodic_pad(self.X_input_2), self.kernal_1, strides=[1, 1, 1, 1], padding='VALID') + self.B_a1
            active_1 = tf.nn.relu(linear_1)
            #conv_1 = tf.nn.avg_pool(active_1, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')
            #conv_1 = tf.nn.dropout(conv_1, keep_prob=keep_prob)

            
            # Configuration of Conv layer 2
            
            k2_S = [3, 3, k_1size, k_2size]
            k2_std = np.sqrt(2)/np.sqrt(np.prod(k2_S[:-1]))
            
            self.kernal_2 = tf.get_variable("W_a2", shape =[3, 3, k_1size, k_2size], initializer = tf.random_normal_initializer(mean=0.0, stddev=k2_std, seed=None))
            self.B_a2 = tf.get_variable("B_a2", shape = [1,1, 1, k_2size], initializer = tf.initializers.zeros())
            linear_2 = tf.nn.conv2d(periodic_pad(active_1), self.kernal_2, strides=[1, 1, 1, 1], padding='VALID') + self.B_a2
            active_2 = tf.nn.relu(linear_2)

            #linear_2 = tf.reshape(linear_2, [1, ]) 

            
            # Configuration of Conv layer 3
            k3_S = [3, 3, k_2size, k_3size]
            k3_std = np.sqrt(1)/np.sqrt(np.prod(k3_S[:-1]))
            
            self.kernal_3 = tf.get_variable("W_a3", shape =[3, 3, k_2size, k_3size], initializer = tf.random_normal_initializer(mean=0.0, stddev=k3_std, seed=None))
            self.B_a3 = tf.get_variable("B_a3", shape = [1,1, 1, k_3size], initializer = tf.initializers.zeros())
            #1D로 Kernal을 만들어야 하니까 [3,3]으로 한다.
            linear_3 = tf.nn.conv2d(periodic_pad(active_2), self.kernal_3, strides=[1, 1, 1, 1], padding='VALID') + self.B_a3
            #active_3 = tf.nn.relu(linear_3)
            #conv_1 = tf.nn.avg_pool(active_1, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')
            #conv_1 = tf.nn.dropout(conv_1, keep_prob=keep_prob)
            #print("linear check")
            #print(linear_3)

            
            linear_4 = tf.reshape(linear_3, [-1, self.output_size_actor[0]*self.output_size_actor[1]*self.output_size_actor[2]])

            output = linear_4 - tf.reduce_mean(linear_4, axis=1, keepdims=True) #(linear_4 - tf.reduce_mean(linear_4, axis=1, keepdims=True)) #그 전에 axis=1로 잘못했었던 것 같은데 (0,1)로 하거나 None으로 했어야 했는데?
             
            self.action_pred = 0.5*output #/tf.math.reduce_std(output, axis=1, keepdims=True)

            
            
            #layer2 = layer2-tf.reduce_mean(layer2, axis=1, keepdims=True)
            #layer2 = 1.0*layer2/tf.reduce_mean(layer2**2, axis=1, keepdims=True)**0.5
            #if action is continuous --> likelihood, discrete --> probability        
            #self.action_pred = layer4 #Action_nums만큼 개수가 추출된다 action list 수

            
            print("Actor_net connected")
            
            
    def initialization_a (self, Objective, name ="ops_name", l_rate=0.00005, B = 0.000001):
        
        self.regular = tf.nn.l2_loss(self.kernal_1) + tf.nn.l2_loss(self.kernal_2) +tf.nn.l2_loss(self.kernal_3) #+ tf.nn.l2_loss(self.kernal_4)
        
        self.Objective = Objective 

        self.actor_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = name)
        #self.actor_vars = [self.kernal_1, self.kernal_2, self.kernal_3, self.kernal_4, self.B_a1, self.B_a2, self.B_a3, self.B_a4] 
        #print(self.actor_vars)          
        self.train = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(-self.Objective, var_list = self.actor_vars)
            

    def predict(self, state): 
        x = np.reshape(state, [1, self.input_size_actor_s[0]*self.input_size_actor_s[1]*self.input_size_actor_s[2]])       
        #[0]: nzp, [1]: nxp, [2]: numbers of state set
        #print("check in actor")
        #print(self.session.run(self.X_input_2, feed_dict={self.X_input: x}))
        return self.session.run(self.action_pred, feed_dict={self.X_input: x})
        #Tensor graph로 짜여있는 self.action_pred는 feed를 받고 (state) 그 값을 내보낸다.

    #Update (학습시키는거 데이터들을 받아서 self.session 실행시켜 돌려보낸다)        
    
    def update(self, critic_net, x_stack):
        #이상태로 그냥 node만 연결시키기는 힘든가보다 graph연결시킬때는 직접 variable을 연결시켜줘야 한다.
        #print("check x_stack")
        #print(x_stack)
        feed = {self.X_input: x_stack, critic_net.input_critic_state: x_stack} 
        #여기서는 action_pred로 이어져 있기때문에 action을 주지 않아도 된다. W_critic은 모두 고정되어 있다.
           
        return self.session.run([self.train], feed_dict=feed)
    
    #state를 넣어줘야 operation들이 돌아가면서 Objective function들이 학습이 된다.