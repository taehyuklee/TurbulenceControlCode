#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from numba import jit
import cupy as cp
import os
import pandas as pd   

#time_step에 대한 기록

#---------- for learning_set path --------------#
time_record_path = 'Monitoring/every_time_recording/'
if not os.path.exists(time_record_path):
    os.makedirs(time_record_path)

Sim_record_path = 'Monitoring/Simulation_result/'
if not os.path.exists(Sim_record_path):
    os.makedirs(Sim_record_path)

Video_path = 'Monitoring/Video_directory/'
if not os.path.exists(Video_path):
    os.makedirs(Video_path)

Contour = 'Monitoring/Contour/'
if not os.path.exists(Contour):
    os.makedirs(Contour)    

Mean_Rms = 'Monitoring/Mean_Rms/'
if not os.path.exists(Mean_Rms):
    os.makedirs(Mean_Rms)    

#---------- for test_set path -------------#
Test_record_path = 'Test_set/Monitoring/every_time_recording/'
if not os.path.exists(Test_record_path):
    os.makedirs(Test_record_path)

Test_Sim_record_path = 'Test_set/Monitoring/Simulation_result/'
if not os.path.exists(Test_Sim_record_path):
    os.makedirs(Test_Sim_record_path)

Test_Video_path = 'Test_set/Monitoring/Video_directory/'
if not os.path.exists(Test_Video_path):
    os.makedirs(Test_Video_path)

Test_Contour = 'Test_set/Monitoring/Contour/'
if not os.path.exists(Test_Contour):
    os.makedirs(Test_Contour)

Test_Mean_Rms = 'Test_set/Monitoring/Mean_Rms/'
if not os.path.exists(Test_Mean_Rms):
    os.makedirs(Test_Mean_Rms)


#------------------------------------------- every time step - Recording part  -------------------------------------------#

def RMS_record(vp_bbc_gpu, t_real, episode, nxp, nzp):

    #Vorticity from gpu to cpu
    vp_bbc = cp.asnumpy(vp_bbc_gpu)
    v_actuator_b = np.real(np.fft.irfft2(vp_bbc, axes=(1,2))*(nxp*nzp)) #dimension 맞춰줘야 한다.


    if episode == "Test":
        RMS_record = open(Test_record_path  + '/RMS file Ep{}.plt'.format(episode), 'a', encoding='utf-8', newline='')
    else:
        RMS_record = open(time_record_path + '/RMS file Ep{}.plt'.format(episode), 'a', encoding='utf-8', newline='')

    if t_real==0:  
        RMS_record.write('VARIABLES = "time", "v_bottom_rms"\n') #, "v_10_rms_bottom"

    RMS_record.write("%d %.5f \n" %(int(t_real), np.std(v_actuator_b[:])))#, np.std(u[32,:,:,1]
    RMS_record.close()


def bulk_record(bulk_v_gpu, t_real, episode):    
    #이 부분에 원래 Initial condition화 되어 있는 속도장을 받아와야한다
    t_real = int(t_real)

    if episode == "Test":
        f_bulk = open(Test_record_path + '/bulk_velocity Ep {}.plt'.format(episode) , 'a', encoding='utf-8', newline='')
    else:
        f_bulk = open(time_record_path + '/bulk_velocity Ep {}.plt'.format(episode) , 'a', encoding='utf-8', newline='')
    
    if t_real==0:      
        f_bulk.write('VARIABLES = "t", "bulk_velocity" \n')	
    
    f_bulk.write('%d %.5f \n' %(int(t_real), bulk_v_gpu))
    f_bulk.close()

def CFL_record(CFL_num_gpu, dt, t_real, episode):    
    #이 부분에 원래 Initial condition화 되어 있는 속도장을 받아와야한다
    t_real = int(t_real)

    if episode == "Test":
        f_CFL = open(Test_record_path + '/CFL num Ep {}.plt'.format(episode) , 'a', encoding='utf-8', newline='')
    else:
        f_CFL = open(time_record_path + '/CFL num Ep {}.plt'.format(episode) , 'a', encoding='utf-8', newline='')
    
    if t_real==0:      
        f_CFL.write('VARIABLES = "t", "CFL_num", "reference" \n')	
    
    f_CFL.write('%d %.5f %.5f\n' %(int(t_real), CFL_num_gpu, 0.5))
    f_CFL.close()

def state_CFL_record(CFL_num_gpu, dt, state_step, t_real, T_state, episode):    
    t_real = int(t_real)

    if episode == "Test":
        f_state_CFL = open('State & episode_unit_record/CFL num (state_unit) Ep {}.plt'.format(episode) , 'a', encoding='utf-8', newline='')
    else:
        f_state_CFL = open('State & episode_unit_record/CFL num (state_unit) Ep {}.plt'.format(episode) , 'a', encoding='utf-8', newline='')
    
    if t_real== (T_state):      
        f_state_CFL.write('VARIABLES = "state_step", "CFL_num", "reference" \n')	
    
    f_state_CFL.write('%d %.5f %.5f\n' %(int(state_step), CFL_num_gpu, 0.5))
    f_state_CFL.close()
    
def reduction_percentage(friction_coff_b_gpu, friction_coff_t_gpu, initial_coeff_gpu, t_real, state_step, episode):    
    #이 부분에 원래 Initial condition화 되어 있는 속도장을 받아와야한다

    if episode == "Test":
        fper = open(Test_record_path + '/drag reduction percentage Ep {}.plt'.format(episode) , 'a',  encoding='utf-8', newline='')	
    else:
        fper = open(time_record_path + '/drag reduction percentage Ep {}.plt'.format(episode) , 'a',  encoding='utf-8', newline='')	
    
    if t_real==0:  
        fper.write('VARIABLES="t", "state_step", "percentage of reduction"\n')
    
    fper.write('%d %d %f \n'%(int(t_real), int(state_step), (friction_coff_b_gpu+friction_coff_t_gpu)/(2*initial_coeff_gpu)*100))
    fper.close()

def state_reduction_percentage(friction_coff_b_gpu, friction_coff_t_gpu, initial_coeff_gpu, initial_coeff_reward_gpu, state_step, t_real, T_state, episode):    
    #이 부분에 원래 Initial condition화 되어 있는 속도장을 받아와야한다

    if episode == "Test":
        f_state_per = open('State & episode_unit_record/drag reduction percentage (state_unit) Ep {}.plt'.format(episode) , 'a',  encoding='utf-8', newline='')	
    else:
        f_state_per = open('State & episode_unit_record/drag reduction percentage (state_unit) Ep {}.plt'.format(episode) , 'a',  encoding='utf-8', newline='')	
    
    if t_real== (T_state):  
        f_state_per.write('VARIABLES="state_step", "percentage of reduction", "friction_coeff_bottom"\n')
    
    bottom = (friction_coff_b_gpu)/(initial_coeff_reward_gpu)*100

    f_state_per.write('%d %.5f %.5f \n'%(int(state_step), (friction_coff_b_gpu+friction_coff_t_gpu)/(2*initial_coeff_gpu)*100, bottom))
    f_state_per.close()


def reduction_percentage_p(t_real, dpdx_mean_gpu, dpdx_mean_init_gpu, state_step, episode):    
    #이 부분에 원래 Initial condition화 되어 있는 속도장을 받아와야한다

    if episode == "Test":
        fper = open(Test_record_path + '/drag reduction percentage Ep {}.plt'.format(episode) , 'a',  encoding='utf-8', newline='')	
    else:
        fper = open(time_record_path + '/drag reduction percentage Ep {}.plt'.format(episode) , 'a',  encoding='utf-8', newline='')	
    
    if t_real==0:  
        fper.write('VARIABLES="t", "state_step", "percentage of reduction (pressure)"\n')

    
    fper.write('%d %d %f \n'%(int(t_real), int(state_step), (dpdx_mean_gpu/dpdx_mean_init_gpu)*100))
    fper.close()


def state_reduction_percentage_p(dpdx_mean_gpu, state_step, t_real, T_state, episode):    
    #이 부분에 원래 Initial condition화 되어 있는 속도장을 받아와야한다

    if episode == "Test":
        f_state_per = open('State & episode_unit_record/drag reduction percentage (state_unit) Ep {}.plt'.format(episode) , 'a',  encoding='utf-8', newline='')	
    else:
        f_state_per = open('State & episode_unit_record/drag reduction percentage (state_unit) Ep {}.plt'.format(episode) , 'a',  encoding='utf-8', newline='')	
    
    if t_real== (T_state):  
        f_state_per.write('VARIABLES="state_step", "percentage of reduction (pressure)"\n')

    initial_dpdx = -1.0
    
    f_state_per.write('%d %.5f %.5f \n'%(int(state_step), dpdx_mean_gpu/(initial_dpdx)*100))

    f_state_per.close()


def Divergence(t_real, div, episode):    
    #이 부분에 원래 Initial condition화 되어 있는 속도장을 받아와야한다
    if episode == "Test":
        f_div = open(Test_record_path + '/Divergence Ep {}.plt'.format(episode) , 'a',  encoding='utf-8', newline='')	
    else:
        f_div = open(time_record_path + '/Divergence Ep {}.plt'.format(episode) , 'a',  encoding='utf-8', newline='')	
    
    if t_real==0:  
        f_div.write('VARIABLES="t", " Divergence"\n')
    
    f_div.write('%d %f \n' %(int(t_real), div))
    f_div.close()


def pressure_gradient(t_real, dpdx_mean_gpu, episode):    
    #이 부분에 원래 Initial condition화 되어 있는 속도장을 받아와야한다
    if episode == "Test":
        f_pg = open(Test_record_path + '/Pressure_gradient Ep {}.plt'.format(episode) , 'a',  encoding='utf-8', newline='')	
    else:
        f_pg = open(time_record_path + '/Pressure_gradient Ep {}.plt'.format(episode) , 'a',  encoding='utf-8', newline='')	
    
    if t_real==0:  
        f_pg.write('VARIABLES="t", " Pressure_gradient"\n')
    
    f_pg.write('%d %f \n' %(int(t_real), dpdx_mean_gpu))
    f_pg.close()


def state_pressure_gradient(t_real, state_step, T_state, dpdx_mean_gpu, past_dpdx_mean_gpu, episode):    
    #이 부분에 원래 Initial condition화 되어 있는 속도장을 받아와야한다
    if episode == "Test":
        f_state_pg = open('State & episode_unit_record/Pressure_gradient (state_unit) Ep {}.plt'.format(episode) , 'a',  encoding='utf-8', newline='')	
    else:
        f_state_pg = open('State & episode_unit_record/Pressure_gradient (state_unit) Ep {}.plt'.format(episode) , 'a',  encoding='utf-8', newline='')	
    
    if t_real== (T_state):  
        f_state_pg.write('VARIABLES="state_unit", "Current pressure_gradient", "Past pressure_gradient"\n')
    
    f_state_pg.write('%d %f %f\n' %(int(state_step), dpdx_mean_gpu, past_dpdx_mean_gpu))
    f_state_pg.close()

    
def simulation_info(t_real, dt, duration, mean_shear_b_gpu ,mean_shear_t_gpu, friction_coff_b_gpu, friction_coff_t_gpu, episode):  #이 부분 
    #이 부분에 원래 Initial condition화 되어 있는 속도장을 받아와야한다
    if episode == "Test":
        f_info = open(Test_record_path + '/simulation_result Ep {}.plt'.format(episode), 'a')
    else:
        f_info = open(time_record_path + '/simulation_result Ep {}.plt'.format(episode), 'a')

    if t_real== 0: 
        f_info.write('VARIABLES="iteration","t","calcuation_time","shear_b","shear_t", "drag_coeff_b", "drag_coeff_t"\n') 
        
    f_info.write('%d %f %f %f %f %f %f \n'%(int(t_real), int(t_real)*dt,duration,mean_shear_b_gpu ,mean_shear_t_gpu, friction_coff_b_gpu, friction_coff_t_gpu))
    f_info.close() 

def reward(long_term, short_term, coef_long_term, coef_short_term, state_step, t_real, T_state, episode):    
    #이 부분에 원래 Initial condition화 되어 있는 속도장을 받아와야한다
    t_real = int(t_real)

    if episode == "Test":
        f_reward = open(Test_record_path + '/reward check Ep {}.plt'.format(episode) , 'a', encoding='utf-8', newline='')
    else:
        f_reward = open(time_record_path + '/reward check Ep {}.plt'.format(episode) , 'a', encoding='utf-8', newline='')
    
    if t_real==  (T_state):      
        f_reward.write('VARIABLES = "t", "state_step", "long_term reward", "short_term reward", "cef_long", "cef_short" \n')	
    
    f_reward.write('%d %d %.5f %.5f %.5f %.5f \n' %(int(t_real), int(state_step), long_term, short_term, coef_long_term, coef_short_term))
    f_reward.close()


#------------------------------------------- Video - Recording part  -------------------------------------------#   
def time_video(up_gpu, next_state_stock, vp_bbc_gpu, vp_tbc_gpu, t_real, episode, nxp, nyp, nzp, dx, dy, dz, y):  

    if t_real%50 ==0 and t_real<=25000: #t_real50 step마다 snapshot을 찍어 영상으로 저장한다

        u = np.zeros([nyp,nzp,nxp,3], dtype=np.float64)
        v_actuator_b  = np.zeros([1,nzp,nxp,1], dtype=np.float64)
        v_actuator_t  = np.zeros([1,nzp,nxp,1], dtype=np.float64)

        state_du_shear_b = next_state_stock[0]
        state_dw_shear_b = next_state_stock[2]
            
        #velocity from gpu to cpu
        up = cp.asnumpy(up_gpu) #to cpu
        u[:,:,:,:] = np.real(np.fft.irfft2(up, axes=(1,2))*(nxp*nzp))
    
        #Vorticity from gpu to cpu
        vp_bbc = cp.asnumpy(vp_bbc_gpu)
        vp_tbc = cp.asnumpy(vp_tbc_gpu)
        v_actuator_b[:,:,:,:] = np.real(np.fft.irfft2(vp_bbc, axes=(1,2))*(nxp*nzp)) #dimension 맞춰줘야 한다.
        v_actuator_t[:,:,:,:] = np.real(np.fft.irfft2(vp_tbc, axes=(1,2))*(nxp*nzp))

        if episode == "Test":        
            f_xy = open(Test_Video_path + "/x_y video Ep {}.plt".format(episode) , 'a', encoding='utf-8', newline='') 
            f_xz = open(Test_Video_path + "/x_z video Ep {}.plt".format(episode) , 'a', encoding='utf-8', newline='')         
            f_yz = open(Test_Video_path + "/y_z video Ep {}.plt".format(episode) , 'a', encoding='utf-8', newline='')   
            f_shear = open(Test_Video_path + "/shear video Ep {}.plt".format(episode) , 'a', encoding='utf-8', newline='')
            f_actuator = open(Test_Video_path + '/Actuator_Video Ep {}.plt'.format(episode), 'a') 
        else:
            f_xy = open(Video_path + "/x_y video Ep {}.plt".format(episode) , 'a', encoding='utf-8', newline='') 
            f_xz = open(Video_path + "/x_z video Ep {}.plt".format(episode) , 'a', encoding='utf-8', newline='')         
            f_yz = open(Video_path + "/y_z video Ep {}.plt".format(episode) , 'a', encoding='utf-8', newline='')   
            f_shear = open(Video_path + "/shear video Ep {}.plt".format(episode) , 'a', encoding='utf-8', newline='')
            f_actuator = open(Video_path + '/Actuator_Video Ep {}.plt'.format(episode), 'a') 

        #-- x-y video --# 
        if t_real==0: 
            f_xy.write('VARIABLES = "x", "y", "u", "v", "w" \n')
    
        f_xy.write('Zone T= "%d"\n' %t_real)  
        f_xy.write('I=%d J=%d\n' %(nxp, nyp))
    
        for j in range(nyp):
            for i in range(nxp):
                f_xy.write('%.5f %.5f %.5f %.5f %.5f \n'%(i*dx,y[j], u[j,nzp//2,i,0],u[j,nzp//2,i,1],u[j,nzp//2,i,2]))
    
        f_xy.close()
    

        #-- x-z video --#
        if t_real==0: 
            f_xz.write('VARIABLES = "x", "z", "u", "v", "w" \n')
    
        f_xz.write('Zone T= "%d"\n' %t_real)  
        f_xz.write('I=%d K=%d\n' %(nxp, nzp))
    
        for k in range(nzp):
            for i in range(nxp):
                f_xz.write('%.5f %.5f %.5f %.5f %.5f \n'%(i*dx, k*dz, u[32,k,i,0],u[32,k,i,1],u[32,k,i,2]))
                #정확히 y+가 얼마인지 알아보자  무차원화 어떻게 되었는지 좀 더 알아보자 

        f_xz.close()    

        #--y-z video --#
        if t_real==0: 
            f_yz.write('VARIABLES = "y", "z", "u", "v", "w" \n')

        f_yz.write('Zone T= "%d"\n' %t_real)  
        f_yz.write('J=%d K=%d\n' %(nyp, nzp))

        for k in range(nzp):
            for j in range(nyp):
                f_yz.write('%.5f %.5f %.5f %.5f %.5f \n'%(y[j], k*dz, u[j,k,nxp//2,0],u[j,k,nxp//2,1],u[j,k,nxp//2,2]))

        f_yz.close()    

        #--shear video --#
        if t_real==0: 
            f_shear.write('VARIABLES = "x", "z", "dudy", "dwdy" \n')

        f_shear.write('Zone T= "%d"\n' %t_real)  
        f_shear.write('I=%d K=%d\n' %(nxp, nzp))

        for k in range(nzp):
            for i in range(nxp):
                f_shear.write('%.5f %.5f %.5f %.5f \n'%(i*dx, k*dz, state_du_shear_b[k,i], state_dw_shear_b[k,i]))

        f_shear.close()   

        #-- blower and Suction profile --#    
        if t_real == 0: 
            f_actuator.write('VARIABLES="x","z","v_bottom", "v_top"\n') 

        f_actuator.write('Zone T="%d"\n' %t_real)
        f_actuator.write('I=%d K=%d\n'%(nxp,nzp))

        for k in range(nzp):
            for i in range(nxp):
                f_actuator.write('%.5f %.5f %.5f %.5f \n' %(i*dx, k*dz, v_actuator_b[0,k,i,0],v_actuator_t[0,k,i,0]))

        f_actuator.close()
        

        
#------------------------------------------- Snap shot - Recording part  -------------------------------------------#
def snap_shot(up_gpu, next_state_stock, shear_wall_dudy, shear_wall_dwdy,  rms_vorticity_gpu, vorticity_gpu, vp_bbc_gpu, vp_tbc_gpu, t_real, episode, nxp, nyp, nzp, dx, dy, dz, y):
    
    #intermediate variables
    mean_u = np.zeros([nyp], dtype= np.float64)
    u = np.zeros([nyp,nzp,nxp,3], dtype=np.float64)
    v_actuator_b  = np.zeros([1,nzp,nxp,1], dtype=np.float64) 
    v_actuator_t  = np.zeros([1,nzp,nxp,1], dtype=np.float64)

    state_du_shear_b = next_state_stock[0]
    state_dw_shear_b = next_state_stock[2]

    
    #from GPU to CPU 
    up = cp.asnumpy(up_gpu) 
    vorticity = cp.asnumpy(vorticity_gpu)
    rms_vorticity = cp.asnumpy(rms_vorticity_gpu)

    #Vorticity from gpu to cpu
    vp_bbc = cp.asnumpy(vp_bbc_gpu)
    vp_tbc = cp.asnumpy(vp_tbc_gpu)
    v_actuator_b[:,:,:,:] = np.real(np.fft.irfft2(vp_bbc, axes=(1,2))*(nxp*nzp)) #dimension 맞춰줘야 한다.
    v_actuator_t[:,:,:,:] = np.real(np.fft.irfft2(vp_tbc, axes=(1,2))*(nxp*nzp)) 
    #또한 [:,:,:,:]로 할당되는건 v_actuator를 앞에 선언하지 않으면 새로 선언되지 않는다.
    
    u[:,:,:,:] = np.real(np.fft.irfft2(up, axes=(1,2))*(nxp*nzp))
    #mean_shear = np.mean((u[1:2,:,:,0:1]-u[0:1,:,:,0:1])/dy[1:2])

    
    #Product of Initialization dataset
    #f_make_init = open('initialization dataset.csv', 'w', encoding='utf-8', newline='')
    #f_make_init.write('"u", "v", "w" \n')

    #vorticity y_z, x_z, x_y
    
    if episode == "Test": 
        #f_vel_3D = open(Test_Sim_record_path + '/CHANNEL time{} Ep {}.plt'.format(int(t_real), episode), 'a')
        #f_vorticity_3D = open(Test_Sim_record_path + '/vorticity time{} Ep {}.plt'.format(int(t_real), episode), 'a')   
        f_mean = open(Test_Mean_Rms + '/y_mean velocity time{} Ep {}.plt'.format(int(t_real), episode), 'a', encoding='utf-8', newline='')
        f_rms = open(Test_Mean_Rms + '/vel_rms fluctuation time{} Ep {}.plt'.format(int(t_real), episode), 'a', encoding='utf-8', newline='') 
        f_vor_rms = open(Test_Mean_Rms + '/vorticity_rms fluctuation time{} Ep {}.plt'.format(int(t_real), episode), 'a', encoding='utf-8', newline='') 
        f_shear = open(Test_Contour +"/shear_field t_real {} Ep {}.plt".format(int(t_real), episode), 'a', encoding='utf-8', newline='') 
        f_shear_wall = open(Test_Contour +"/shear_field_wall t_real {} Ep {}.plt".format(int(t_real), episode), 'a', encoding='utf-8', newline='') 

        #Contour for y+ = 10 location (y-z plane)
        f_vorticity_contour_yz = open(Test_Contour +'/vorticity_contour (y-z) t{} Ep {}.plt'.format(int(t_real), episode), 'a')
        f_contour_vel_yz = open(Test_Contour + "/contour_vel (y-z) t{} Ep {}.plt".format(int(t_real), episode) , 'a', encoding='utf-8', newline='') 

        #Contour for y+ = 10 location (x-z plane)
        f_contour_vel_xz = open(Test_Contour +'/contour_vel (x-z) t{} Ep {}.plt'.format(int(t_real), episode), 'a')
        f_contour_act_xz = open(Test_Contour +'/contour_act (x-z) t{} Ep {}.plt'.format(int(t_real), episode), 'a')
        f_contour_act_xz_scatter = open(Test_Contour +'/contour_act (x-z) scatter_t{} Ep {}.csv'.format(int(t_real), episode), 'a') #for scatter plot
        f_vorticity_contour_xz = open(Test_Contour +'/vorticity_contour (x-z) t{} Ep {}.plt'.format(int(t_real), episode), 'a')

        #Contour for (x-y plane)
        f_contour_vel_xy = open(Test_Contour + "/contour_vel (x-y) t{} Ep {}.plt".format(int(t_real), episode) , 'a', encoding='utf-8', newline='') 
        f_vorticity_contour_xy = open(Test_Contour +'/vorticity_contour (x-y) t{} Ep {}.plt'.format(int(t_real), episode), 'a')

        #mid action
        f_actor_mid_line_yz = open(Test_Mean_Rms + '/Action_mid line y-z t{} Ep {}.plt'.format(int(t_real), episode), 'a', encoding='utf-8', newline='')
        f_actor_mid_line_xy = open(Test_Mean_Rms + '/Action_mid line x-y t{} Ep {}.plt'.format(int(t_real), episode), 'a', encoding='utf-8', newline='')

        #mid action
        f_actor_mid_line_yz_scat = open(Test_Mean_Rms + '/Action_mid scatter y-z t{} Ep {}.csv'.format(int(t_real), episode), 'a', encoding='utf-8', newline='')
        f_actor_mid_line_xy_scat = open(Test_Mean_Rms + '/Action_mid scatter x-y t{} Ep {}.csv'.format(int(t_real), episode), 'a', encoding='utf-8', newline='')


    else:
        #f_vel_3D = open(Sim_record_path + '/CHANNEL time{} Ep {}.plt'.format(int(t_real), episode), 'a')
        #f_vorticity_3D = open(Sim_record_path + '/vorticity time{} Ep {}.plt'.format(int(t_real), episode), 'a')   
        f_mean = open(Mean_Rms + '/y_mean velocity time{} Ep {}.plt'.format(int(t_real), episode), 'a', encoding='utf-8', newline='')
        f_rms = open(Mean_Rms + '/y_rms fluctuation time{} Ep {}.plt'.format(int(t_real), episode), 'a', encoding='utf-8', newline='') 
        f_vor_rms = open(Mean_Rms + '/vorticity_rms fluctuation time{} Ep {}.plt'.format(int(t_real), episode), 'a', encoding='utf-8', newline='') 
        f_shear = open(Contour +"/shear_field t_real {} Ep {}.plt".format(int(t_real), episode), 'a', encoding='utf-8', newline='') 
        f_shear_wall = open(Contour +"/shear_field_wall t_real {} Ep {}.plt".format(int(t_real), episode), 'a', encoding='utf-8', newline='') 


        #Contour for y+ = 10 location (y-z plane)
        f_vorticity_contour_yz = open(Contour +'/vorticity_contour (y-z) t{} Ep {}.plt'.format(int(t_real), episode), 'a')
        f_contour_vel_yz = open(Contour + "/contour_vel (y-z) t{} Ep {}.plt".format(int(t_real), episode) , 'a', encoding='utf-8', newline='') 

        #Contour for y+ = 10 location (x-z plane)
        f_contour_vel_xz = open(Contour +'/contour_vel (x-z) t{} Ep {}.plt'.format(int(t_real), episode), 'a')
        f_contour_act_xz = open(Contour +'/contour_act (x-z) t{} Ep {}.plt'.format(int(t_real), episode), 'a')
        f_contour_act_xz_scatter = open(Contour +'/contour_act_scatter (x-z) t{} Ep {}.plt'.format(int(t_real), episode), 'a')
        f_vorticity_contour_xz = open(Contour +'/vorticity_contour (x-z) t{} Ep {}.plt'.format(int(t_real), episode), 'a')

        #Contour for (x-y plane)
        f_contour_vel_xy = open(Contour + "/contour_vel (x-y) t{} Ep {}.plt".format(int(t_real), episode) , 'a', encoding='utf-8', newline='') 
        f_vorticity_contour_xy = open(Contour + '/vorticity_contour (x-y) t{} Ep {}.plt'.format(int(t_real), episode), 'a')

        f_actor_mid_line_yz = open(Mean_Rms +'/Action_mid line y-z t{} Ep {}.plt'.format(int(t_real), episode), 'a', encoding='utf-8', newline='')
        f_actor_mid_line_xy = open(Mean_Rms +'/Action_mid line x-y t{} Ep {}.plt'.format(int(t_real), episode), 'a', encoding='utf-8', newline='')

        f_actor_mid_line_yz_scat = open(Mean_Rms +'/Action_mid scatter y-z t{} Ep {}.csv'.format(int(t_real), episode), 'a', encoding='utf-8', newline='')
        f_actor_mid_line_xy_scat= open(Mean_Rms +'/Action_mid scatter x-y t{} Ep {}.csv'.format(int(t_real), episode), 'a', encoding='utf-8', newline='')

    '''
    #3D velocity channel
    f_vel_3D.write('VARIABLES="x","y","z","u","v","w"\n') 
    f_vel_3D.write('Zone T="HIT%d"\n'%0)
    f_vel_3D.write('I=%d J=%d K=%d\n'%(nxp,nyp,nzp))

    #Vorticity file recording (3D) 
    f_vorticity_3D.write('VARIABLES="x","y","z","w_x","w_y","w_z"\n')
    f_vorticity_3D.write('Zone T="HIT%d"\n'%0)
    f_vorticity_3D.write('I=%d J=%d K=%d\n'%(nxp-1,nyp-1,nzp-1))
    '''

    #statistics (Mean & RMS) of Channel flow
    #Writing each variables
    f_mean.write('VARIABLES = "y", "u_mean" \n')	
    f_rms.write('VARIABLES = "y", "u(y)_rms", "v(y)_rms", "w(y)_rms" \n')
    f_vor_rms.write('VARIABLES = "y", "vor(x)_rms", "vor(y)_rms", "vor(z)_rms" \n')

    #Shear field
    f_shear.write('VARIABLES = "x", "z", "dudy", "dwdy" \n')
    f_shear.write('Zone T= "%d"\n' %t_real)  
    f_shear.write('I=%d K=%d\n' %(nxp, nzp))

    #Shear_wall field
    f_shear_wall.write('VARIABLES = "x", "z", "dudy", "dwdy" \n')
    f_shear_wall.write('Zone T= "%d"\n' %t_real)  
    f_shear_wall.write('I=%d K=%d\n' %(nxp, nzp))


    #writing each variables (x-z plane)
    f_contour_vel_xz.write('VARIABLES = "x", "z", "u", "v", "w" \n')  #x-z 방향 u,v,w velocity
    f_contour_vel_xz.write('Zone T="HIT%d"\n'%0)
    f_contour_vel_xz.write('I=%d K=%d\n' %(nxp, nzp))

    f_contour_act_xz.write('VARIABLES="x","z","v_bottom", "v_top"\n')  # Actuator 
    f_contour_act_xz.write('Zone T="HIT%d"\n'%0)
    f_contour_act_xz.write('I=%d K=%d\n' %(nxp, nzp))

    #for scatter plot
    f_contour_act_xz_scatter.write('VARIABLES="x","z","v_bottom", "v_top"\n') 


    f_vorticity_contour_xz.write('VARIABLES="x","z","w_x","w_y","w_z","u", "v", "w"\n')  # vorticity
    f_vorticity_contour_xz.write('Zone T="HIT%d"\n'%0)
    f_vorticity_contour_xz.write('I=%d K=%d\n' %(nxp, nzp))
 

    #writing each variables (y-z plane)


    f_vorticity_contour_yz.write('VARIABLES="y","z","w_x","w_y","w_z","u", "v", "w"\n')  # vorticity
    f_vorticity_contour_yz.write('Zone T="HIT%d"\n'%0)
    f_vorticity_contour_yz.write('J=%d K=%d\n' %(nyp-1, nzp))


    f_contour_vel_yz.write('VARIABLES = "y", "z", "u", "v", "w" \n')  #x-z 방향 u,v,w velocity
    f_contour_vel_yz.write('Zone T="HIT%d"\n'%0)
    f_contour_vel_yz.write('J=%d K=%d\n' %(nyp, nzp))


    #writing each variables (x-y plane)

    f_vorticity_contour_xy.write('VARIABLES="x","y","w_x","w_y","w_z","u", "v", "w"\n')  # vorticity
    f_vorticity_contour_xy.write('Zone T="HIT%d"\n'%0)
    f_vorticity_contour_xy.write('I=%d J=%d\n' %(nxp, nyp-1))
    
    f_contour_vel_xy.write('VARIABLES = "x", "y", "u", "v", "w" \n')  #x-z 방향 u,v,w velocity
    f_contour_vel_xy.write('Zone T="HIT%d"\n'%0)
    f_contour_vel_xy.write('I=%d K=%d\n' %(nxp, nyp))

    f_actor_mid_line_yz.write('VARIABLES = "z", "action"\n')
    f_actor_mid_line_xy.write('VARIABLES = "x", "action"\n')

    f_actor_mid_line_yz_scat.write('VARIABLES = "z", "action"\n')
    f_actor_mid_line_xy_scat.write('VARIABLES = "x", "action"\n')



    #mean & RMS
    mean_u = np.mean(u[:,:,:,0], axis=(1,2))  #axis - - x, z축에 대해서만 평균을 내고싶다는 것
    rms_u = np.std(u[:,:,:,:], axis=(1,2))  #axis #axis - x, z축에 대해



    #product initialization dataset
    '''
    for i in range(nxp):
        for k in range(nzp):
            for j in range(nyp):
                f_make_init.write('%.5f, %.5f, %.5f \n' %(u[j,k,i,0],u[j,k,i,1],u[j,k,i,2]))
    
    #Channel velocity 3-D
    for k in range(nzp):
        for j in range(nyp):
            for i in range(nxp):
                f_vel_3D.write('%.5f %.5f %.5f %.5f %.5f %.5f \n'%(i*dx,y[j],k*dz, u[j,k,i,0],u[j,k,i,1],u[j,k,i,2]))
    '''    
    #RMS (part)
    for j in range(nyp):
        f_mean.write('%.5f %.5f \n' %(y[j], mean_u[j]))    
        f_rms.write('%.5f  %.5f  %.5f  %.5f \n' %(y[j], rms_u[j,0], rms_u[j,1], rms_u[j,2]))

    #Vorticity RMS 
    for j in range(nyp-1):
        f_vor_rms.write('%.5f  %.5f  %.5f  %.5f \n' %((y[j]+y[j+1])/2, rms_vorticity[j,0], rms_vorticity[j,1], rms_vorticity[j,2]))

    #actuator contour
    for k in range(nzp):
        for i in range(nxp):
            f_contour_act_xz.write('%.5f, %.5f, %.5f, %.5f \n' %(i*dx, k*dz, v_actuator_b[0,k,i,0],v_actuator_t[0,k,i,0]))

    #actuator contour_scatter
    for k in range(nzp):
        for i in range(nxp):
            f_contour_act_xz_scatter.write('%.5f, %.5f, %.5f, %.5f \n' %(i*dx, k*dz, v_actuator_b[0,k,i,0],v_actuator_t[0,k,i,0]))

    #velocity of y+10 location (x-z plane)
    for k in range(nzp):
        for i in range(nxp):
            f_contour_vel_xz.write('%.5f %.5f %.5f %.5f %.5f \n'%(i*dx, k*dz, u[32,k,i,0],u[32,k,i,1],u[32,k,i,2]))

    #velocity at the location (y-z plane)
    for k in range(nzp):
        for j in range(nyp):
            f_contour_vel_yz.write('%.5f %.5f %.5f %.5f %.5f \n'%(y[j], k*dz, u[j,k,nxp//2,0],u[j,k,nxp//2,1],u[j,k,nxp//2,2]))


    #velocity at the location (x-y plane)
    for j in range(nyp):
        for i in range(nxp):
            f_contour_vel_xy.write('%.5f %.5f %.5f %.5f %.5f \n'%(i*dx, y[j], u[j,nzp//2,i,0],u[j,nzp//2,i,1],u[j,nzp//2,i,2]))
      
    
    '''
    #Vorticity 3-D 
    for k in range(nzp-1):
        for j in range(nyp-1):
            for i in range(nxp-1):
                f_vorticity_3D.write('%.5f %.5f %.5f %.5f %.5f %.5f \n'%(i*dx,y[j],k*dz, vorticity[j,k,i,0],vorticity[j,k,i,1],vorticity[j,k,i,2]))
    '''        
    
    #Vorticity Contour x-z
    for k in range(nzp):
        for i in range(nxp):
            f_vorticity_contour_xz.write('%.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f\n'%(i*dx ,k*dz, vorticity[32,k,i,0],vorticity[32,k,i,1],vorticity[32,k,i,2],u[32,k,i,0],u[32,k,i,1],u[32,k,i,2]))


    #Vorticity Contour y-z
    for k in range(nzp):
        for j in range(nyp-1):
            f_vorticity_contour_yz.write('%.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f\n'%((y[j]+y[j])/2,k*dz,  vorticity[j,k,nxp//2,0],vorticity[j,k,nxp//2,1],vorticity[j,k,nxp//2,2],u[j,k,nxp//2,0],u[j,k,nxp//2,1],u[j,k,nxp//2,2]))

    #Vorticity Contour x-y
    for j in range(nyp-1):
        for i in range(nxp):
            f_vorticity_contour_xy.write('%.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f\n'%(i*dx, (y[j]+y[j])/2, vorticity[j,nzp//2,i,0],vorticity[j,nzp//2,i,1],vorticity[j,nzp//2,i,2],u[j,nzp//2,i,0],u[j,nzp//2,i,1],u[j,nzp//2,i,2]))


    #shear stress#
    for k in range(nzp):
        for i in range(nxp):
            f_shear.write('%.5f %.5f %.5f %.5f\n'%(i*dx, k*dz, state_du_shear_b[k,i], state_dw_shear_b[k,i]))
    
    #shear stress_wall#
    for k in range(nzp):
        for i in range(nxp):
            f_shear_wall.write('%.5f %.5f %.5f %.5f\n'%(i*dx, k*dz, shear_wall_dudy[0,k,i,0],shear_wall_dwdy[0,k,i,0]))       

    for k in range(nzp):
        f_actor_mid_line_yz.write('%.5f %.5f \n'%((k*dz, v_actuator_b[0,k,nxp//2,0])))

    for i in range(nxp):
        f_actor_mid_line_xy.write('%.5f %.5f \n'%((i*dx, v_actuator_b[0,nzp//2,i,0])))

    for k in range(nzp):
        f_actor_mid_line_yz_scat.write('%.5f, %.5f \n'%((k*dz, v_actuator_b[0,k,nxp//2,0])))

    for i in range(nxp):
        f_actor_mid_line_xy_scat.write('%.5f, %.5f \n'%((i*dx, v_actuator_b[0,nzp//2,i,0])))


    #f_make_init.close()
    #f_vel_3D.close()
    f_mean.close()
    f_rms.close()
    #f_vorticity_3D.close()
    f_contour_vel_xz.close()
    f_contour_vel_xy.close()
    f_contour_vel_yz.close()
    f_contour_act_xz.close()
    f_contour_act_xz_scatter.close()
    f_vorticity_contour_xz.close()
    f_vorticity_contour_yz.close()
    f_vorticity_contour_xy.close()
    f_shear.close()
    f_shear_wall.close()
    f_vor_rms.close()
    f_actor_mid_line_yz.close()
    f_actor_mid_line_xy.close()
    f_actor_mid_line_yz_scat.close()
    f_actor_mid_line_xy_scat.close()


def save_3D_field(up_gpu,episode,t_real,nxp, nyp, nzp):
    
    #intermediate variables
    u = np.zeros([nyp,nzp,nxp,3], dtype=np.float64)

    #from GPU to CPU 
    up = cp.asnumpy(up_gpu) 
  
    u[:,:,:,:] = np.real(np.fft.irfft2(up, axes=(1,2))*(nxp*nzp))


    f_make_init = open('mid dataset ep{} t_real{}.csv'.format(episode, t_real), 'w', encoding='utf-8', newline='')
    f_make_init.write('"u", "v", "w" \n')

    for i in range(nxp):
        for k in range(nzp):
            for j in range(nyp):
                f_make_init.write('%.5f, %.5f, %.5f \n' %(u[j,k,i,0],u[j,k,i,1],u[j,k,i,2]))


    f_make_init.close()





def Scatter_plot(up_gpu, next_state_stock, vp_bbc_gpu, vp_tbc_gpu, shear_wall_dudy, shear_wall_dwdy, t_real, episode, nxp, nyp, nzp, dx, dy, dz, y, Re):

    #intermediate variables
    mean_u = np.zeros([nyp, nzp, nxp,1], dtype= np.float64)
    u = np.zeros([nyp,nzp,nxp,3], dtype=np.float64)
    dudy = np.zeros([nyp,nzp,nxp,1], dtype=np.float64)
    v_actuator_b  = np.zeros([1,nzp,nxp,1], dtype=np.float64) 
    v_actuator_t  = np.zeros([1,nzp,nxp,1], dtype=np.float64)
    Covariance_coef_u  = np.zeros([nyp,3], dtype=np.float64)
    Covariance_coef_dudy  = np.zeros([nyp,1], dtype=np.float64)
    Covariance_coef_shear  = np.zeros([nyp,3], dtype=np.float64)
    Covariance_coef_shear_du  = np.zeros([nyp,1], dtype=np.float64)
    #for correlation between each layers
    shaer_layer_u = np.zeros([nyp-1, nzp, nxp,1], dtype = np.float64)
    shaer_layer_w = np.zeros([nyp-1, nzp, nxp,1], dtype = np.float64)
    Correlation_layers_u = np.zeros([nyp], dtype=np.float64)
    Correlation_layers_w = np.zeros([nyp], dtype=np.float64)


    state_du_shear_b = next_state_stock[0]
    state_dw_shear_b = next_state_stock[2]

    #from GPU to CPU 
    up = cp.asnumpy(up_gpu) 
    #vorticity = cp.asnumpy(vorticity_gpu)

    #Vorticity from gpu to cpu
    vp_bbc = cp.asnumpy(vp_bbc_gpu)
    vp_tbc = cp.asnumpy(vp_tbc_gpu)
    v_actuator_b[:,:,:,:] = np.real(np.fft.irfft2(vp_bbc, axes=(1,2))*(nxp*nzp)) #dimension 맞춰줘야 한다.
    v_actuator_t[:,:,:,:] = np.real(np.fft.irfft2(vp_tbc, axes=(1,2))*(nxp*nzp)) 
    #또한 [:,:,:,:]로 할당되는건 v_actuator를 앞에 선언하지 않으면 새로 선언되지 않는다.
    
    u[:,:,:,:] = np.real(np.fft.irfft2(up, axes=(1,2))*(nxp*nzp))
    dudy[1:nyp-1,:,:,:] = 0.5*(u[1:nyp-1,:,:,0:1]-u[0:nyp-2,:,:,0:1])/(dy[1:nyp-1]) \
                         +0.5*(u[2:nyp,:,:,0:1]-u[1:nyp-1,:,:,0:1])/(dy[2:nyp])
    dudy[0:1,:,:,:] = (u[1:2,:,:,0:1]-u[0:1,:,:,0:1])/(dy[1:2])
    dudy[nyp-1:nyp,:,:,:] = (u[nyp-1:nyp,:,:,0:1]-u[nyp-2:nyp-1,:,:,0:1])/(dy[nyp-1:nyp])

    scatter_plot = open(Test_Sim_record_path + '/scatter_plot time{} Ep {}.plt'.format(int(t_real), episode), 'a', encoding='utf-8', newline='')
    scatter_plot.write('VARIABLES = "input_dudy", "input_dwdy","v_bottom", "v_top","u", "v", "w","shear_dudy","shear_dwdy"\n') 

    scatter_csv = open(Test_Sim_record_path + '/scatter_csv time{} Ep {}.csv'.format(int(t_real), episode), 'a', encoding='utf-8', newline='')
    scatter_csv.write('VARIABLES = "input_dudy", "input_dwdy","v_bottom","v_top", "u", "v", "w","shear_dudy","shear_dwdy"\n') 

    RMS_text = open(Test_Sim_record_path + '/RMS_text file time{} Ep {}.text'.format(int(t_real), episode), 'a', encoding='utf-8', newline='')
    RMS_text.write('VARIABLES = "v_bottom_rms"\n') #, "v_10_rms_bottom"
    RMS_text.write("%f \n" %(np.std(v_actuator_b[:])))#, np.std(u[32,:,:,1]
    RMS_text.close()

    for j in range(nyp):
        for pp in range(3):
            Covariance_coef_u[j,pp] = np.mean((v_actuator_b[0,:,:,0] - np.mean(v_actuator_b[0,:,:,0]))*(u[j,:,:,pp] - np.mean(u[j,:,:,pp])))/(np.std(v_actuator_b[0,:,:,0])*np.std(u[j,:,:,pp]))
            Covariance_coef_shear[j,pp] = np.mean((state_du_shear_b[:,:] - np.mean(state_du_shear_b[:,:]))*(u[j,:,:,pp] - np.mean(u[j,:,:,pp])))/(np.std(state_du_shear_b[:,:])*np.std(u[j,:,:,pp]))
        Covariance_coef_dudy[j,0] = np.mean((v_actuator_b[0,:,:,0] - np.mean(v_actuator_b[0,:,:,0]))*(dudy[j,:,:,0] - np.mean(dudy[j,:,:,0])))/(np.std(v_actuator_b[0,:,:,0])*np.std(dudy[j,:,:,0]))
        Covariance_coef_shear_du[j,0] = np.mean((state_du_shear_b[:,:] - np.mean(state_du_shear_b[:,:]))*(dudy[j,:,:,0] - np.mean(dudy[j,:,:,0])))/(np.std(state_du_shear_b[:,:])*np.std(dudy[j,:,:,0]))


    Covariance_coef_u[np.isnan(Covariance_coef_u)] = 0
    Covariance_coef_dudy[np.isnan(Covariance_coef_dudy)] = 0
    Covariance_coef_shear[np.isnan(Covariance_coef_shear)] = 0
    Covariance_coef_shear_du[np.isnan(Covariance_coef_shear_du)] = 0

    correlation_record = open(Test_Sim_record_path + '/Correlation V_actuator time {}.plt'.format(int(t_real)), 'a', encoding='utf-8', newline='')  
    correlation_record.write('VARIABLES= "y", "coef_u", "coef_v", "coef_w", "coef_dudy" \n')  
    correlation_record.write('Zone T="zone%d"\n'%int(t_real))

    correlation_record_shear = open(Test_Sim_record_path + '/Correlation shear time {}.plt'.format(int(t_real)), 'a', encoding='utf-8', newline='')  
    correlation_record_shear.write('VARIABLES= "y", "coef_u", "coef_v", "coef_w", "coef_dudy" \n')  
    correlation_record_shear.write('Zone T="zone%d"\n'%int(t_real))


    for j in range(nyp):
        correlation_record.write("%f %f %f %f %f\n" %(y[j], Covariance_coef_u[j,0], Covariance_coef_u[j,1], Covariance_coef_u[j,2], Covariance_coef_dudy[j,0]))
        correlation_record_shear.write("%f %f %f %f %f\n" %(y[j], Covariance_coef_shear[j,0], Covariance_coef_shear[j,1], Covariance_coef_shear[j,2], Covariance_coef_shear_du[j,0]))

    #layers shear
    corre_layer_record = open(Test_Sim_record_path + '/Correlation shear_layers du time {}.plt'.format(int(t_real)), 'a', encoding='utf-8', newline='')  
    corre_layer_record.write('VARIABLES= "y", "coef_du_shear"\n')  
    corre_layer_record.write('Zone T="zone%d"\n'%int(t_real))   


    #wall correlation (shear stress correlation between layers)
    for j in range(nyp-1):
        shaer_layer_u[j:j+1,:,:,:] = (u[j+1:j+2,:,:,0:1] - u[j:j+1,:,:,0:1])/(Re*dy[j+1:j+2])
        shaer_layer_w[j:j+1,:,:,:] = (u[j+1:j+2,:,:,2:3] - u[j:j+1,:,:,2:3])/(Re*dy[j+1:j+2])

    for j in range(nyp-2): #두개씩 짝을 지으니까 한개 더 주네
        Correlation_layers_u[j] = np.mean((shaer_layer_u[j,:,:,:] - np.mean(shaer_layer_u[j,:,:,:]))*(shaer_layer_u[j+1,:,:,:] - np.mean(shaer_layer_u[j+1,:,:,:])))/(np.std(shaer_layer_u[j,:,:,:])*np.std(shaer_layer_u[j+1,:,:,:]))
        Correlation_layers_w[j] = np.mean((shaer_layer_w[j,:,:,:] - np.mean(shaer_layer_w[j,:,:,:]))*(shaer_layer_w[j+1,:,:,:] - np.mean(shaer_layer_w[j+1,:,:,:])))/(np.std(shaer_layer_w[j,:,:,:])*np.std(shaer_layer_w[j+1,:,:,:]))
        corre_layer_record.write("%f %f %f\n" %(y[j], Correlation_layers_u[j], Correlation_layers_w[j]))



    #dudy & v_bottom profile , dwdy & v_bottom profile, u_velocity & v_bottom profile, shear field 2.1 & shear field 0
    for k in range(nzp):
        for i in range(nxp):
            scatter_plot.write('%.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f\n'%(state_du_shear_b[k,i], state_dw_shear_b[k,i], v_actuator_b[0,k,i,0], v_actuator_t[0,k,i,0], u[32,k,i,0],u[32,k,i,1],u[32,k,i,2], shear_wall_dudy[0,k,i,0], shear_wall_dwdy[0,k,i,0]))
            scatter_csv.write('%.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f\n'%(state_du_shear_b[k,i], state_dw_shear_b[k,i], v_actuator_b[0,k,i,0],v_actuator_t[0,k,i,0], u[32,k,i,0],u[32,k,i,1],u[32,k,i,2], shear_wall_dudy[0,k,i,0], shear_wall_dwdy[0,k,i,0]))

    scatter_plot.close()
    scatter_csv.close()
    correlation_record.close()
    correlation_record_shear.close()
    corre_layer_record.close()
###############################################################################################################################################################