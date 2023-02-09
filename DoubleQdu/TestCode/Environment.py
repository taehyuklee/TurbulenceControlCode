#!/usr/bin/env python
# coding: utf-8

# In[4]:

#Channel environment
import numpy as np
from numba import jit
import cupy as cp
import random
import os
import pandas as pd
import monitoring as monitor
from timeit import default_timer as timer    

Test_time_avg = 'Test_set/Monitoring/time vaerage/'
if not os.path.exists(Test_time_avg):
    os.makedirs(Test_time_avg)

# ---------------------- Global method ---------------------#
def TDMA_gpu(a,b,c,d,w,g,p,n):

    w[0] = c[0]/b[0]
    g[0] = d[0]/b[0]

    for i in range(1,n-1):
        w[i] = c[i]/(b[i] - a[i]*w[i-1])
    for i in range(1,n):
        g[i] = (d[i] - a[i]*g[i-1])/(b[i] - a[i]*w[i-1])
    p[n-1] = g[n-1]
    for i in range(n-1,0,-1):
        p[i-1] = g[i-1] - w[i-1]*p[i]
    return p

def zero_padding(up):
    up[:,:,nx//2:,:] = 0; up[:,nz//2:nzp-(nz//2-1),:,:] = 0

def CFL (up_storage_gpu, dx, dy, dz, dt, CFL_num_gpu):

    CFL_num_gpu = dt*cp.max(cp.abs(up_storage_gpu[:,:,:,0:1]/dx) + cp.abs(up_storage_gpu[:,:,:,1:2]/(0.5*dy[0:nyp,0:1,0:1,0:1] + 0.5*dy[1:nyp+1,0:1,0:1,0:1])) + cp.abs(up_storage_gpu[:,:,:,2:3]/dz))

    return CFL_num_gpu

#-----------------------------------------------------------#    


#Simulation parameter
pi = np.pi
xl = 4.0*pi; yl = 2.0; zl = 2.0*pi
nxp = 128; nyp = 193; nzp = 128  #nyp = 129
nx = int(nxp/3*2); ny = nyp; nz = int(nzp/3*2)
dx = xl/nxp; dz = zl/nzp
Re = 180.0
Pr = 0.71
dt = 0.0001
wall = 9
starting_avg = 100000
time_avg_snap = 2000
#실제로는 +3 위치임
#10 위치 y+ ~ 1.6, 11 위치 y+ ~1.87 12 위치 y+ ~ 2.1

#real time (time_step for simulation)
t_real = np.zeros([1], dtype=np.int32)

#T_state_up
T_state_up = np.zeros([1], dtype=np.int32)

############################################################ Global variables declare ###################################################
#time average할수 있도록 놓는 부분 중간 중간 statistics

#참고로 Global variable말고 Class안에서 생성자로 선언했으면 좀 더 깔끔했겠지만, 앞에 self.를 전체에 붙여야해서 그냥 Global로 함#
u_t_avg = np.zeros([nyp], dtype=np.float64); rms_t_avg = np.zeros([nyp, 3], dtype=np.float64); Re_t_avg = np.zeros([1], dtype=np.float64)
mean_u = np.zeros([nyp], dtype=np.float64); rms_u = np.zeros([nyp, 3], dtype=np.float64)
bulk_v = np.zeros([1], dtype=np.float64)
bulk_v_init = np.zeros([1], dtype=np.float64)
vp_bbc = np.zeros([1,nzp,nxp//2+1,1], dtype=np.complex128); vp_tbc = np.zeros([1,nzp,nxp//2+1,1], dtype=np.complex128)
friction_coff_b = np.zeros([1], dtype=np.float64); friction_coff_t = np.zeros([1], dtype=np.float64)
initial_coeff = np.zeros([1], dtype=np.float64)
vorticity = np.zeros([nyp-1,nzp-1,nxp-1,3], dtype=np.float64)
pressure = np.zeros([nyp,nzp,nxp,1], dtype=np.float64)
initial_coeff_reward = np.zeros([1], dtype=np.float64); past_friction_reward = np.zeros([1], dtype=np.float64)
dpdx_mean_init = np.zeros([1], dtype=np.float64); dpdx_mean = np.zeros([1], dtype=np.float64); past_dpdx_mean = np.zeros([1], dtype=np.float64)

vorticity_p = np.zeros([nyp-1,nzp,(nxp)//2+1,3], dtype=np.complex128)
vorticity = np.zeros([nyp-1,nzp,(nxp),3], dtype=np.float64)
vorticity_t_avg = np.zeros([nyp-1, 3], dtype=np.float64)

v_actuator_b  = np.zeros([1,nzp,nxp,1], dtype=np.complex128)
v_actuator_t  = np.zeros([1,nzp,nxp,1], dtype=np.complex128)
u = np.zeros([nyp,nzp,nxp,3], dtype=np.float64)
u_storage = np.zeros([nyp,nzp,nxp,3], dtype=np.float64)
up_mid = np.zeros([nyp,nzp,nxp//2+1,3], dtype=np.complex128)
T = np.zeros([nyp,nzp,nxp,1], dtype=np.float64)
CFL_num = np.zeros([1], dtype=np.float64)


#for application
w_action_b = np.zeros([1,nzp,nxp,1], dtype = np.float64)
wf_action_b = np.zeros([1,nzp,nxp//2+1,1], dtype = np.complex128)

w_action_t = np.zeros([1,nzp,nxp,1], dtype = np.float64)
wf_action_t = np.zeros([1,nzp,nxp//2+1,1], dtype = np.complex128)



dudx = np.zeros([nyp,nzp,nxp,3], dtype=np.float64)
dudy = np.zeros([nyp,nzp,nxp,3], dtype=np.float64)
dudz = np.zeros([nyp,nzp,nxp,3], dtype=np.float64)
H1p = np.zeros([nyp-2,nzp,nxp//2+1,3], dtype=np.complex128)
H2p = np.zeros([nyp-2,nzp,nxp//2+1,3], dtype=np.complex128)

pp = np.zeros([nyp+1,nzp,nxp//2+1,1], dtype=np.complex128)
divp = np.zeros([nyp,nzp,nxp//2+1,1], dtype=np.complex128)

#dy (Chevyshev nodes calculation)
y = np.zeros([nyp,1,1,1], dtype=np.float64)
dy = np.zeros([nyp+1,1,1,1], dtype=np.float64)

for j in range(nyp):#
    y[j,:,:,:] = -np.tanh(2.5*(1.0-2.*j/(nyp-1)))/np.tanh(2.5) #y[j,:,:,:] = -np.cos(float(j)/float(nyp-1)*pi)#-1 + j*2.0/(nyp-1)
dy[1:nyp] = y[1:nyp,:,:,:] - y[0:nyp-1,:,:,:]; dy[0:1] = dy[1:2]; dy[nyp:nyp+1] = dy[nyp-1:nyp]


#wave number들을 넣어주는 부분
alpha = np.zeros([1,1,nxp//2+1,1], dtype=np.float64)
for k in range(nxp//2+1):
    alpha[0,0,k,0] = 2*pi/xl*k

gamma = np.zeros([1,nzp,1,1], dtype=np.float64)
for k in range(nzp//2):
    gamma[0,k,0,0] = 2*pi/zl*k
    gamma[0,nzp-k-1,0,0] = 2*pi/zl*(-k-1)
    
#CPU --> GPU로 각 변수들에 대해 넘겨주는 부분
up_mid_gpu = cp.asarray(up_mid) #up_mid는 projection method에서 압력 term을 update하기 전
dy_gpu = cp.asarray(dy) #dy는 함수화한거기때문에 gpu로 넘겨줘서 계산해야함
alpha_gpu = cp.asarray(alpha); gamma_gpu = cp.asarray(gamma)
H1p_gpu = cp.asarray(H1p); H2p_gpu = cp.asarray(H2p); pp_gpu = cp.asarray(pp)
dudx_gpu = cp.asarray(dudx); dudy_gpu = cp.asarray(dudy); dudz_gpu = cp.asarray(dudz)

wf_action_b_gpu = cp.asarray(wf_action_b)
w_action_b = cp.asarray(w_action_b)

wf_action_t_gpu = cp.asarray(wf_action_t)
w_action_t = cp.asarray(w_action_t)

#TDMA Veloctiy coefficient
a = np.zeros([nyp-2,1,1,1], dtype=np.complex128); b = np.zeros([nyp-2,1,1,1], dtype=np.complex128)
c = np.zeros([nyp-2,1,1,1], dtype=np.complex128); d = np.zeros([nyp-2,nzp,nxp//2+1,3], dtype=np.complex128)
a_gpu = cp.asarray(a); b_gpu = cp.asarray(b); c_gpu = cp.asarray(c); d_gpu = cp.asarray(d)
wtdma = np.zeros([nyp-3,nzp,nxp//2+1,3], dtype=np.complex128); gtdma = np.zeros([nyp-2,nzp,nxp//2+1,3], dtype=np.complex128)
ptdma = np.zeros([nyp-2,nzp,nxp//2+1,3], dtype=np.complex128)
wtdma_gpu = cp.asarray(wtdma); gtdma_gpu = cp.asarray(gtdma); ptdma_gpu = cp.asarray(ptdma)

#TDMA pressure coefficient
app = np.zeros([nyp-1,nzp,nxp//2+1,1], dtype=np.complex128); bpp = np.zeros([nyp-1,nzp,nxp//2+1,1], dtype=np.complex128)
cpp = np.zeros([nyp-1,nzp,nxp//2+1,1], dtype=np.complex128); dpp = np.zeros([nyp-1,nzp,nxp//2+1,1], dtype=np.complex128)
ap_gpu = cp.asarray(app); bp_gpu = cp.asarray(bpp); cp_gpu = cp.asarray(cpp); dp_gpu = cp.asarray(dpp)
wtdma_p = np.zeros([nyp-2,nzp,nxp//2+1,1], dtype=np.complex128); gtdma_p = np.zeros([nyp-1,nzp,nxp//2+1,1], dtype=np.complex128)
ptdma_p = np.zeros([nyp-1,nzp,nxp//2+1,1], dtype=np.complex128)
wtdma_p_gpu = cp.asarray(wtdma_p); gtdma_p_gpu = cp.asarray(gtdma_p); ptdma_p_gpu = cp.asarray(ptdma_p)


#velocity TDMA coefficient
a_gpu = -dt/(2*Re)/dy_gpu[1:nyp-1]/(0.5*dy_gpu[1:nyp-1]+0.5*dy_gpu[2:nyp])
b_gpu = 1.0+dt/(2*Re)*alpha_gpu**2+dt/(2*Re)*gamma_gpu**2+dt/(2*Re)*(1.0/dy_gpu[1:nyp-1]+1.0/dy_gpu[2:nyp])/(0.5*dy_gpu[1:nyp-1]+0.5*dy_gpu[2:nyp])
c_gpu = -dt/(2*Re)/dy_gpu[2:nyp]/(0.5*dy_gpu[1:nyp-1]+0.5*dy_gpu[2:nyp])

#pressure TDMA coefficient
ap_gpu[0:nyp-1,:,:] = 1.0/dy_gpu[1:nyp,:,:]/(0.5*dy_gpu[0:nyp-1,:,:]+0.5*dy_gpu[1:nyp,:,:])
bp_gpu[0:nyp-1,:,:] = -1.0*alpha_gpu**2-1.0*gamma_gpu**2-1.0/dy_gpu[1:nyp,:,:]*(1.0/(0.5*dy_gpu[0:nyp-1,:,:]+0.5*dy_gpu[1:nyp,:,:])
                                                                               +1.0/(0.5*dy_gpu[1:nyp,:,:]+0.5*dy_gpu[2:nyp+1,:,:]))
cp_gpu[0:nyp-1,:,:] = 1.0/dy_gpu[1:nyp,:,:]/(0.5*dy_gpu[1:nyp,:,:]+0.5*dy_gpu[2:nyp+1,:,:])

bp_gpu[0:1] = -1.0*alpha_gpu**2-1.0*gamma_gpu**2-1.0*(1.0/dy_gpu[1:2,:,:])/(0.5*dy_gpu[1:2,:,:]+0.5*dy_gpu[2:3,:,:]) #boundary 부분
bp_gpu[nyp-2:nyp-1] = -1.0*alpha_gpu**2-1.0*gamma_gpu**2-1.0*(1.0/dy_gpu[nyp-1:nyp,:,:])/(0.5*dy_gpu[nyp-2:nyp-1,:,:]+0.5*dy_gpu[nyp-1:nyp,:,:]) 

#To avoid singularity of TDMA matix
bp_gpu[:1,0:1,0:1,:] = 1; cp_gpu[:1,0:1,0:1,:] = 0; dp_gpu[:1,0:1,0:1,:] = 0


#Real space to Fourier domain
up = np.fft.rfft2(u, axes=(1,2))/(nzp*nxp) #왜 nzp, nxp 격자 scale로 나누는거지?
zero_padding(up)


## Cpu to GPU ##
up_gpu = cp.asarray(up); up_storage_gpu = cp.asarray(u_storage); 
u_t_avg_gpu = cp.asarray(u_t_avg); rms_t_avg_gpu = cp.asarray(rms_t_avg); Re_t_avg_gpu = cp.asarray(Re_t_avg)
mean_u_gpu = cp.asarray(mean_u); rms_u_gpu = cp.asarray(rms_u)
bulk_v_gpu = cp.asarray(bulk_v)
bulk_v_init_gpu = cp.asarray(bulk_v_init)
vp_bbc_gpu = cp.asarray(vp_bbc); vp_tbc_gpu = cp.asarray(vp_tbc)
friction_coff_b_gpu = cp.asarray(friction_coff_b); friction_coff_t_gpu = cp.asarray(friction_coff_t)
initial_coeff_gpu = cp.asarray(initial_coeff)
vorticity_gpu = cp.asarray(vorticity)
initial_coeff_reward_gpu = cp.asarray(initial_coeff_reward); past_friction_reward_gpu = cp.asarray(past_friction_reward)
dpdx_mean_init_gpu = cp.asarray(dpdx_mean_init); dpdx_mean_gpu = cp.asarray(dpdx_mean); past_dpdx_mean_gpu = cp.asarray(past_dpdx_mean)
CFL_num_gpu = cp.asarray(CFL_num)

vorticity_gpu = cp.asarray(vorticity)
vorticity_t_avg_gpu = cp.asarray(vorticity_t_avg) 
vorticity_p_gpu = cp.asarray(vorticity_p)
###################################################################################################################################################

#---------- formation of stack for two-point correlation ------------#
R_11_stream = np.zeros([nxp], dtype=np.float64); R_11_span = np.zeros([nzp], dtype=np.float64)
R_22_stream = np.zeros([nxp], dtype=np.float64); R_22_span = np.zeros([nzp], dtype=np.float64)
R_33_stream = np.zeros([nxp], dtype=np.float64); R_33_span = np.zeros([nzp], dtype=np.float64)

u_fluc = np.zeros([1,nzp,nxp], dtype=np.float64)
v_fluc = np.zeros([1,nzp,nxp], dtype=np.float64)
w_fluc = np.zeros([1,nzp,nxp], dtype=np.float64)

R_11_stream_gpu = cp.asarray(R_11_stream); R_11_span_gpu = cp.asarray(R_11_span)
R_22_stream_gpu = cp.asarray(R_22_stream); R_22_span_gpu = cp.asarray(R_22_span)
R_33_stream_gpu = cp.asarray(R_33_stream); R_33_span_gpu = cp.asarray(R_33_span)

u_fluc_gpu = cp.asarray(u_fluc) 
v_fluc_gpu = cp.asarray(v_fluc) 
w_fluc_gpu = cp.asarray(w_fluc)

fluc_nums = 15000
f_t = 0

u_fluc_stack = np.zeros([fluc_nums,nzp,nxp], dtype = np.float64)
v_fluc_stack = np.zeros([fluc_nums,nzp,nxp], dtype = np.float64)
w_fluc_stack = np.zeros([fluc_nums,nzp,nxp], dtype = np.float64)

u_fluc_stack_gpu = cp.asarray(u_fluc_stack)
v_fluc_stack_gpu = cp.asarray(v_fluc_stack)
w_fluc_stack_gpu = cp.asarray(w_fluc_stack)
#---------------------------------------------------------------------#
    
#Environment class    
class env(): 

    def reset(self, episode): #초기화하고 state를 보내준다
        
        ##################### Global varible for Channel flow ######################
        global u_vel, v_vel, z_vel, u, up_mid_gpu, up, up_gpu, pp_gpu, pressure_gpu, t_real, yl, dt, wall
        global a_gpu, b_gpu, c_gpu, d_gpu, wtdma_gpu, gtdma_gpu, ptdma_gpu, bulk_v_gpu, bulk_v_init_gpu
        global ap_gpu, bp_gpu, cp_gpu, dp_gpu, wtdma_p_gpu, gtdma_p_gpu, ptdma_p_gpu, mean_u_gpu
        global Re_t_avg_gpu,rms_t_avg_gpu, u_t_avg_gpu, vorticity_t_avg_gpu, vorticity_gpu, vorticity_p_gpu
        global nxp, nyp, nzp, dy_gpu, dz, dx, alpha_gpu, gamma_gpu
        global H1p_gpu, H2p_gpu, dudx_gpu, dudy_gpu, dudz_gpu
        global up_gpu, up_storage_gpu, up_mid_gpu, vp_bbc_gpu, vp_tbc_gpu
        global nxp, nyp, nzp, dy_gpu, dz, dx, alpha_gpu, gamma_gpu, t_real, Action_num
        global dpdx_mean_gpu, dpdx_mean_init_gpu, friction_coff_b_gpu, friction_coff_t_gpu
        ############################################################################
        
        #다시 초기의 Fully developed channel condition으로 돌아간다 #
        if episode == "Test":
            #data_frame_init = pd.read_csv(r"/home/fish991/code/Half_staggered/Test_folder/Test_dudw/mid dataset/mid dataset epTest t_real10000.csv", encoding = "ISO-8859-1")
            data_frame_init = pd.read_csv(r"/home/fish991/code/Half_staggered/Env_wide/developed.csv", encoding = "ISO-8859-1")


        da = np.array(data_frame_init)
        #data_int = np.reshape(d_array_init, (3,3,3))
        
        #u velocity structure (y, z, x 순서로 되어 있다)
        u_vel = np.transpose(np.reshape(da[:,0], (nxp,nzp,nyp)), (2,1,0))
        #u velocity structure (y, z, x 순서로 되어 있다)
        v_vel = np.transpose(np.reshape(da[:,1], (nxp,nzp,nyp)), (2,1,0))
        #u velocity structure (y, z, x 순서로 되어 있다)
        z_vel = np.transpose(np.reshape(da[:,2], (nxp,nzp,nyp)), (2,1,0))
        
        #u_velocity (y, z, x 순서로 되어 있다)
        u[:,:,:,0] = u_vel
        #v velocity (y, z, x 순서로 되어 있다)
        u[:,:,:,1] = v_vel
        #w velocity (y, z, x 순서로 되어 있다)
        u[:,:,:,2] = z_vel

        #form CPU to GPU (only velocity u) so as to get shear stress
        u_gpu = cp.asarray(u)

        #Initial bulk velocity
        mean_u_gpu = cp.mean(u_gpu[:,:,:,0], axis=(1,2))
        bulk_v_init_gpu = cp.sum(mean_u_gpu[:] * (0.5*dy_gpu[0:nyp,0,0,0] + 0.5*dy_gpu[1:nyp+1,0,0,0])) #cp.sum(up_gpu[1:nyp-1,0:1,0:1,0:1] * (0.5*dy_gpu[0:nyp,0,0,0] + 0.5*dy_gpu[1:nyp+1,0,0,0]))
        bulk_v_init_gpu = bulk_v_init_gpu/(float(yl))  

        #Real space to Fourier domain in order to reset 
        up = np.fft.rfft2(u, axes=(1,2))/(nzp*nxp) #왜 nzp, nxp 격자 scale로 나누는거지?
        zero_padding(up)
        
        #CPU to GPU (아래 Simulator를 위해 up_gpu초기화 시켜놓는 부분)
        up_gpu = cp.asarray(up)

        #time step initialization
        t_real = int(0)

        ################# One time step for dp/dx #########################
        up_gpu[0:1] = 0.0; up_gpu[nyp-1:nyp] = 0.0; 

        vp_bbc_gpu[:] = 0 #action_output_gpu_b #-up_gpu[7:8,:,:,1:2]
        vp_tbc_gpu[:] = 0 #-action_output_gpu_t  #-up_gpu[nyp-1-7:nyp-7,:,:,1:2]

        up_gpu[0:1,:,:,1:2] = vp_bbc_gpu[:] #up는 fourier domain에 있다.
        up_gpu[nyp-1:nyp,:,:,1:2] = vp_tbc_gpu[:]

        #zero_padding(up_gpu)
        zero_padding(up_gpu)

        #Get velocity and gradient of velocity (physical space)
        dudxp_gpu = 1j*alpha_gpu*up_gpu[:,:,:,0:3]; dudzp_gpu = 1j*gamma_gpu*up_gpu[:,:,:,0:3]
        u_gpu = cp.real(cp.fft.irfft2(up_gpu, axes=(1,2))*(nzp*nxp))
        dudx_gpu = cp.real(cp.fft.irfft2(dudxp_gpu, axes=(1,2))*(nzp*nxp))
        dudz_gpu = cp.real(cp.fft.irfft2(dudzp_gpu, axes=(1,2))*(nzp*nxp))
        #dudy는 FDM을 사용할 것이기때문에 Fourier domain이 아닌 그대로 차분식을 사용한다. 
        dudy_gpu[1:nyp-1] = 0.5*(u_gpu[1:nyp-1]-u_gpu[0:nyp-2])/(dy_gpu[1:nyp-1]) \
                           +0.5*(u_gpu[2:nyp]-u_gpu[1:nyp-1])/(dy_gpu[2:nyp])

        #nonlinear term momentum
        H1p_gpu[:] = H2p_gpu[:]
        #그러네 nonlinear term을 다 배분해서 풀어서 쓰지 않고 그 앞에 만들어 놓은 derivative term을 그대로 사용하여 만들었다. 훨씬 효율적이네 
        H2_gpu = u_gpu[1:nyp-1,:,:,0:1]*dudx_gpu[1:nyp-1] + u_gpu[1:nyp-1,:,:,1:2]*dudy_gpu[1:nyp-1] + u_gpu[1:nyp-1,:,:,2:3]*dudz_gpu[1:nyp-1] 
        #convective term
        
        #H2_gpu[:,:,:,0:1] += -1 #<dp/dx>=-1 --> 이게 뭐지?
        H2p_gpu[:] = cp.fft.rfft2(H2_gpu, axes=(1,2))/(nxp*nzp)
        
        if t_real == 0: H1p_gpu[:] = H2p_gpu[:]
        
        #velocity TDMA
        d_gpu = ((1.0-dt/(2*Re)*alpha_gpu**2-dt/(2*Re)*gamma_gpu**2)*up_gpu[1:nyp-1]
            +dt/(2*Re)*(+(up_gpu[2:nyp]-up_gpu[1:nyp-1])/dy_gpu[2:nyp]
                        -(up_gpu[1:nyp-1]-up_gpu[0:nyp-2])/dy_gpu[1:nyp-1])
                        /(0.5*dy_gpu[1:nyp-1]+0.5*dy_gpu[2:nyp])
            +dt*(-1.5*H2p_gpu[0:nyp-2]+0.5*H1p_gpu[0:nyp-2])) #이 부분에 boundary condition 들어가야한다.
            #-1.5에서 보면 Adams bashforth방법을 사용하였다. 

        d_gpu[0:1,:,:,1:2] += - a_gpu[0:1] * vp_bbc_gpu[:]
        d_gpu[nyp-3:nyp-2,:,:,1:2] += - c_gpu[nyp-3:nyp-2] * vp_tbc_gpu[:]
        
        
        up_mid_gpu[1:nyp-1,:,:,:] = TDMA_gpu(a_gpu,b_gpu,c_gpu,d_gpu,wtdma_gpu,gtdma_gpu,ptdma_gpu,nyp-2) 
        up_mid_gpu[0:1] = 0.0 #여기 바뀌어야 함
        up_mid_gpu[nyp-1:nyp] = 0.0 #여기도 바뀌어야 함
        up_mid_gpu[0:1,:,:,1:2] = vp_bbc_gpu[:]
        up_mid_gpu[nyp-1:nyp,:,:,1:2] = vp_tbc_gpu[:]
        
        #pressure TDMA
        dp_gpu[0:nyp-1,:,:] = 1.0/dt*(+1j*alpha_gpu*(0.5*up_mid_gpu[0:nyp-1,:,:,0:1]+0.5*up_mid_gpu[1:nyp,:,:,0:1]) 
                                        +1j*gamma_gpu*(0.5*up_mid_gpu[0:nyp-1,:,:,2:3]+0.5*up_mid_gpu[1:nyp,:,:,2:3])
                                        +(up_mid_gpu[1:nyp,:,:,1:2]-up_mid_gpu[0:nyp-1,:,:,1:2])/dy_gpu[1:nyp])
        

        pp_gpu[1:nyp,:,:,:] = TDMA_gpu(ap_gpu,bp_gpu,cp_gpu,dp_gpu,wtdma_p_gpu,gtdma_p_gpu,ptdma_p_gpu,nyp-1)
        
        pp_gpu[0:1,:,:] = pp_gpu[1:2,:,:]
        pp_gpu[nyp:nyp+1,:,:] = pp_gpu[nyp-1:nyp,:,:]
          
        #get pressure gradient
        dpdxp_gpu = 1j*alpha_gpu*(0.5*pp_gpu[0:nyp]+0.5*pp_gpu[1:nyp+1])
        dpdzp_gpu = 1j*gamma_gpu*(0.5*pp_gpu[0:nyp]+0.5*pp_gpu[1:nyp+1])                     
        dpdyp_gpu = (pp_gpu[1:nyp+1]-pp_gpu[0:nyp])/(0.5*dy_gpu[1:nyp+1]+0.5*dy_gpu[0:nyp])
        
        ##mean pressure gradient (<dpdx> = -1) --> nonlinear term
        #dpdxp_gpu[:,0:1,0:1] = dpdxp_gpu[:,0:1,0:1] - 1
        
        # pressure correction (pressure term에 의한 update 부분)
        up_gpu[1:nyp-1,:,:,0:1] = up_mid_gpu[1:nyp-1,:,:,0:1] - dt * dpdxp_gpu[1:nyp-1]
        up_gpu[1:nyp-1,:,:,1:2] = up_mid_gpu[1:nyp-1,:,:,1:2] - dt * dpdyp_gpu[1:nyp-1]
        up_gpu[1:nyp-1,:,:,2:3] = up_mid_gpu[1:nyp-1,:,:,2:3] - dt * dpdzp_gpu[1:nyp-1]

        zero_padding(up_gpu)

        up_storage_gpu[:,:,:,:] = cp.real(np.fft.irfft2(up_gpu, axes=(1,2))*(nxp*nzp))

        #Constant flow 유량 일정하게 다시 바꿔주는 부분
        mean_u_gpu = cp.mean(up_storage_gpu[:,:,:,0], axis=(1,2))
        bulk_v_gpu = cp.sum(mean_u_gpu[:] * (0.5*dy_gpu[0:nyp,0,0,0] + 0.5*dy_gpu[1:nyp+1,0,0,0])) #cp.sum(up_gpu[1:nyp-1,0:1,0:1,0:1] * (0.5*dy_gpu[0:nyp,0,0,0] + 0.5*dy_gpu[1:nyp+1,0,0,0]))
        bulk_v_gpu = bulk_v_gpu/(float(yl))  
        dpdx_mean_gpu = (bulk_v_gpu - bulk_v_init_gpu) / dt #for moment reward
        dpdx_mean_init_gpu =  -1.0 #(bulk_v_gpu - bulk_v_init_gpu) / dt #for long term reward

        print("reset dpdx_mean")
        print(dpdx_mean_gpu)
        
        up_gpu[1:nyp-1,0:1,0:1,0:1] = up_gpu[1:nyp-1,0:1,0:1,0:1] - dt * cp.reshape(dpdx_mean_gpu, [1,1,1,1])
        

        #------------------------------------------------------------- Pressure graidnet 한 번 구해주기 ----------------------------------------------------------------------#

        #(Calculation) Initial state (dw/dy, du/dy, pressure)
        mean_shear_b_gpu_init = cp.real(cp.mean((up_gpu[1:2,0:1,0:1,0:1]-up_gpu[0:1,0:1,0:1,0:1])/dy_gpu[1:2]))
        mean_shear_t_gpu_init = cp.real(cp.mean((up_gpu[nyp-2:nyp-1,0:1,0:1,0:1]-up_gpu[nyp-1:nyp,0:1,0:1,0:1])/dy_gpu[nyp-1:nyp]))


        #Shear stress (wall velocity gradient) y, z, x (in order)
        #(du/dy)
        shear_b_gpu_du = (u_gpu [4 + wall:5 + wall,:,:,0:1]-u_gpu [3 + wall:4 + wall,:,:,0:1])/dy_gpu[4 + wall:5 + wall]
        shear_t_gpu_du = (u_gpu[nyp-4-wall:nyp-3-wall,:,:,0:1]-u_gpu[nyp-5-wall:nyp-4-wall,:,:,0:1])/dy_gpu[nyp-4-wall:nyp-3-wall]

        #(dw/dy)
        shear_b_gpu_dw = (u_gpu [4+wall:5+wall,:,:,2:3]-u_gpu [3+wall:4+wall,:,:,2:3])/dy_gpu[4+wall:5+wall]
        shear_t_gpu_dw = (u_gpu[nyp-4-wall:nyp-3-wall,:,:,2:3]-u_gpu[nyp-5-wall:nyp-4-wall,:,:,2:3])/dy_gpu[nyp-4-wall:nyp-3-wall]

        #initial friction (reset)
        friction_coff_b_gpu = mean_shear_b_gpu_init*2.0/(180*bulk_v_gpu**2)
        friction_coff_t_gpu = mean_shear_t_gpu_init*2.0/(180*bulk_v_gpu**2)

        #real time reset (to zero)
        t_real = np.zeros([1], dtype=np.float64) #요소 하나인데 잘못 하고 있었다 바꿔라

        #Gpu to CPU in order to use states in control algorithm
        shear_b_du= cp.asnumpy(shear_b_gpu_du); shear_t_du= cp.asnumpy(shear_t_gpu_du)
        shear_b_dw= cp.asnumpy(shear_b_gpu_dw); shear_t_dw= cp.asnumpy(shear_t_gpu_dw)
        #pressure_b = pressure_state[1, :, :, :]; pressure_t = pressure_state[nyp-1, :, :, :]

        #print("check shape again")
        #print(shear_b_du[0, :, :, :].shape) #[1, 96, 96, 1] 이 형태로 나오네

        self.state_stock = (shear_b_du[0, :, :, 0]/Re, shear_t_du[0, :, :, 0]/Re, shear_b_dw[0, :, :, 0]/Re, shear_t_dw[0, :, :, 0]/Re) 
        #차원 축소 Image에 필요한 부분들 nzp, nxp가 필요함 each channel
        self.state_stock = list(self.state_stock)
        #(shear_b_gpu_du, shear_t_gpu_du, shear_b_gpu_dw, shear_t_gpu_dw, pp_gpu)











        ########################################## 한 step의 속도장 계산은 여기서 끝난다 ############################################
            
        #Shear stress (wall velocity gradient)
        mean_shear_b_gpu = cp.real(cp.mean((up_gpu[1:2,0:1,0:1,0:1]-up_gpu[0:1,0:1,0:1,0:1])/dy_gpu[1:2]))
        mean_shear_t_gpu = cp.real(cp.mean((up_gpu[nyp-2:nyp-1,0:1,0:1,0:1]-up_gpu[nyp-1:nyp,0:1,0:1,0:1])/dy_gpu[nyp-1:nyp]))

            
        #Fourier domain to Real domain (to obtain statsitcs)
        up_storage_gpu[:,:,:,:] = cp.real(np.fft.irfft2(up_gpu, axes=(1,2))*(nxp*nzp))
            
        # Vorticity 구해주는 부분 #
            
        #vorticity (x-direction) d(v_z)/dy - d(v_y)/dz
        vorticity_p_gpu[:,:,:,0:1] = (up_gpu[1:nyp,:,:,2:3] - up_gpu[0:nyp-1,:,:,2:3])/dy_gpu[0:nyp-1,:, :,:] \
                            - (1j*gamma_gpu*(up_gpu[0:nyp-1,:,:,1:2] + up_gpu[1:nyp,:,:,1:2])/2)

        #vorticity (y-direction) d(v_x)/dz - d(v_z)/dx
        vorticity_p_gpu[:,:,:,1:2] = 1j*gamma_gpu*((up_gpu[0:nyp-1,:,:,0:1] + up_gpu[1:nyp,:,:,0:1])/2) - 1j*alpha_gpu*((up_gpu[0:nyp-1,:,:,2:3] + up_gpu[1:nyp,:,:,2:3])/2)

        #vorticity (z-direction) d(v_y)/dx - d(v_x)/dy
        vorticity_p_gpu[:,:,:,2:3] = 1j*alpha_gpu*((up_gpu[0:nyp-1,:,:,1:2] + up_gpu[1:nyp,:,:,1:2])/2) \
                        - (up_gpu[1:nyp,:,:,0:1] - up_gpu[0:nyp-1,:,:,0:1])/dy_gpu[0:nyp-1,:, :,:]

        vorticity_gpu[:,:,:,:] = cp.real(cp.fft.irfft2(vorticity_p_gpu, axes=(1,2))*(nxp*nzp))  



        #average & std 구해주는 부분#
        mean_u_gpu = cp.mean(up_storage_gpu[:,:,:,0], axis=(1,2)) #mean velocity profile
        rms_u_gpu = cp.std(up_storage_gpu[:,:,:,:], axis=(1,2)) #RMS (x, z에 대해서는 고정해놓고 y에 대해서만)
        rms_vorticity_gpu = cp.std(vorticity_gpu[:,:,:,:], axis=(1,2))   
        
        
        #Bulk Velocity 유량 바꿔주는 부분#
        bulk_v_gpu = cp.sum(mean_u_gpu[:] * (0.5*dy_gpu[0:nyp,0,0,0] + 0.5*dy_gpu[1:nyp+1,0,0,0]))
        bulk_v_gpu = bulk_v_gpu/(float(yl))          
        bulk_v_return = cp.asnumpy(bulk_v_gpu) 

        #Drag Coefficient 구하는 부분#
        friction_coff_b_gpu = mean_shear_b_gpu*2.0/(180*bulk_v_gpu**2)
        friction_coff_t_gpu = mean_shear_t_gpu*2.0/(180*bulk_v_gpu**2)

        if t_real == 0: #(t_real은 진짜 시간이다)
                initial_coeff_gpu  =(friction_coff_b_gpu + friction_coff_t_gpu)/2


        #(du/dy)
        shear_b_next_du_gpu = (up_storage_gpu[4+wall:5+wall,:,:,0:1] - up_storage_gpu[3+wall:4+wall,:,:,0:1])/dy_gpu[4+wall:5+wall]
        shear_t_next_du_gpu = (up_storage_gpu[nyp-4-wall:nyp-3-wall,:,:,0:1] - up_storage_gpu[nyp-5-wall:nyp-4-wall,:,:,0:1])/dy_gpu[nyp-4-wall:nyp-3-wall]

        #(dw/dy)
        shear_b_next_dw_gpu = (up_storage_gpu[4+wall:5+wall,:,:,2:3] -up_storage_gpu[3+wall:4+wall,:,:,2:3])/dy_gpu[4+wall:5+wall]
        shear_t_next_dw_gpu = (up_storage_gpu[nyp-4-wall:nyp-3-wall,:,:,2:3] - up_storage_gpu[nyp-5-wall:nyp-4-wall,:,:,2:3])/dy_gpu[nyp-4-wall:nyp-3-wall]            

        #(pressure)
        shear_b_next_du= cp.asnumpy(shear_b_next_du_gpu); shear_t_next_du= cp.asnumpy(shear_t_next_du_gpu)
        shear_b_next_dw= cp.asnumpy(shear_b_next_dw_gpu); shear_t_next_dw= cp.asnumpy(shear_t_next_dw_gpu)

        #(du/dy - wall)
        shear_wall_dudy = (up_storage_gpu[1:2,:,:,0:1] - up_storage_gpu[0:1,:,:,0:1])/(Re*dy_gpu[1:2])

        #(dw/dy - wall)
        shear_wall_dwdy = (up_storage_gpu[1:2,:,:,2:3] - up_storage_gpu[0:1,:,:,2:3])/(Re*dy_gpu[1:2])

        next_state_stock = (shear_b_next_du[0, :, :, 0]/Re, shear_t_next_du[0, :, :, 0]/Re, shear_b_next_dw[0, :, :, 0]/Re, shear_t_next_dw[0, :, :, 0]/Re)
        #[nzp, nxp] 필요한 정보만 따로 빼내어 주도록 하자 
        next_state_stock = list(next_state_stock)
        self.next_state_b[:,:,0] = next_state_stock[0]#; self.next_state_b[:,:,1] = next_state_stock[2]
        self.next_state_t[:,:,0] = next_state_stock[1]#; self.next_state_t[:,:,1] = next_state_stock[3]

        self.state_interval_b = np.reshape(self.next_state_b, [1, self.state_shape[0]*self.state_shape[1]*self.state_shape[2]])
        self.state_interval_t = np.reshape(self.next_state_t, [1, self.state_shape[0]*self.state_shape[1]*self.state_shape[2]])

        episode = "Initial_field"
        monitor.snap_shot(up_gpu, next_state_stock, shear_wall_dudy, shear_wall_dwdy, rms_vorticity_gpu, vorticity_gpu, vp_bbc_gpu, vp_tbc_gpu, t_real, episode, nxp, nyp, nzp, dx, dy, dz, y)





















        
        return self.state_stock

     

    def state_num (self): #state 개수를 돌려주는 역할을 한다
        
        global nxp, nzp

        state_num = 3*nzp*nxp  #state_1.shape[0] #+ state_2.shape[0] #+ state_3.shape[0]
        self.state_channel =1
        self.state_shape = [nzp, nxp, self.state_channel]

        #self.state_interval = np.zeros([1, nzp*nxp*self.state_channel], dtype = np.float64) #for simulation iteration
        self.next_state_b = np.zeros([nzp, nxp, self.state_channel], dtype = np.float64)
        self.next_state_t = np.zeros([nzp, nxp, self.state_channel], dtype = np.float64)        

        return state_num, self.state_shape

    
    
    def action_setting (self): #action에 대한 개수를 return해 주는 역할을 한다
        
        global nxp, nzp
            
        Action_num = nzp*nxp
        self.action_channel = 1

        action_shape = [nzp, nxp, self.action_channel]

        return Action_num, action_shape






    def test_performance(self, main_actor, end_step, episode_t, video_record_freq_t, snap_shot_freq_t,scatter_freq): 
    
        global nxp, nyp, nzp, dy_gpu, dz, dx, alpha_gpu, gamma_gpu, t_real, Action_num, yl, wall
        global up_gpu, up_storage_gpu, up_mid_gpu, vp_bbc_gpu, vp_tbc_gpu
        global friction_coff_b_gpu , friction_coff_t_gpu, initial_coeff_gpu, vorticity_gpu, pp_gpu
        global H1p_gpu, H2p_gpu, dudx_gpu, dudy_gpu, dudz_gpu
        global a_gpu, b_gpu, c_gpu, d_gpu, wtdma_gpu, gtdma_gpu, ptdma_gpu
        global ap_gpu, bp_gpu, cp_gpu, dp_gpu, wtdma_p_gpu, gtdma_p_gpu, ptdma_p_gpu
        global mean_u_gpu, rms_u_gpu, bulk_v_gpu, bulk_v_init_gpu, initial_coeff_reward_gpu, past_friction_reward_gpu  #statistics 
        global dpdx_mean_gpu, CFL_num_gpu
        global vorticity_gpu, vorticity_t_avg_gpu, vorticity_p_gpu
        global R_11_stream_gpu, R_22_stream_gpu, R_33_stream_gpu, u_fluc_gpu, v_fluc_gpu, w_fluc_gpu, u_fluc_stack_gpu, v_fluc_stack_gpu, w_fluc_stack_gpu, f_t

        bulk_v_return = np.zeros([1], dtype=np.float64)

        t_real = int(t_real)
        state_step = 0

        if t_real ==0: #top 부분 state받아오 주는거 <맨 위에 reset부분에서 self.state_stock 생성자로 해서 받아와준다# main에서 받아오는건 bottom만 데려온다
            #Simulation part에서는 이렇게 bottom부분 처리를 안했는데, main으로 bottom state를 빼주기 위해서 main에서 state_b형태로 주는 것이었다
            self.next_state_b[:,:,0] = self.state_stock[0]#; self.next_state_b[:,:,1] = self.state_stock[2]
            self.state_interval_b = np.reshape(self.next_state_b, [1, self.state_shape[0]*self.state_shape[1]*self.state_shape[2]])            
            
            self.next_state_t[:,:,0] = self.state_stock[1]#; self.next_state_t[:,:,1] = self.state_stock[3]
            self.state_interval_t = np.reshape(self.next_state_t, [1, self.state_shape[0]*self.state_shape[1]*self.state_shape[2]])

        for t in range(int(end_step)): 

            #bottom plate actuator
            action_output_b = main_actor.predict(self.state_interval_b) #이 부분 잘못됨 
            action_output_b = np.reshape(action_output_b, (1, nzp, nxp, self.action_channel)) 
            action_output_gpu_b = cp.asarray(action_output_b)
            action_output_gpu_b = cp.fft.rfft2(action_output_gpu_b, axes=(1,2))/(nxp*nzp)

            #top plate actuator
            action_output_t = main_actor.predict(-self.state_interval_t)  #top은 Initialization하면서 받아오는걸로 class내에서 생성자로 받아온다.
            action_output_t = np.reshape(action_output_t, (1, nzp, nxp, self.action_channel)) 
            action_output_gpu_t = cp.asarray(action_output_t)
            action_output_gpu_t = cp.fft.rfft2(action_output_gpu_t, axes=(1,2))/(nxp*nzp)

            start = timer()   
    
            zero_padding(up_gpu)
            #up_gpu[0:1] = 0.0; up_gpu[nyp-1:nyp] = 0.0; 


            #----------------------------------------------------- dwdy control (Deterministic) -----------------------------------------------------#
            '''
            #wf_action_b_gpu[:] = 1j*gamma_gpu[:,:,:,:]*(up_gpu[1:2,:,:,2:3] - up_gpu[0:1,:,:,2:3])/dy_gpu[1:2,:,:,:]/abs(gamma_gpu[:,:,:,:])
            wf_action_b_gpu[:] = 1j*gamma_gpu[:,:,:,:]*(up_gpu[12:13,:,:,2:3] - up_gpu[11:12,:,:,2:3])/dy_gpu[12:13,:,:,:]/abs(gamma_gpu[:,:,:,:])
            wf_action_b_gpu[:,0,:] = 0.0
            #A = (up_gpu[1:2,:,:,2:3] - up_gpu[0:1,:,:,2:3])/(dy_gpu[1:2,:,:,:]))
            #print(wf_action_b_gpu[:])
            #print(abs(gamma_gpu))
            w_action_b[:] = cp.real(cp.fft.irfft2(wf_action_b_gpu, axes=(1,2))*(nzp*nxp))

            #wf_action_t_gpu[:,:,:] = 1j*gamma_gpu[:,:,:,:]*(up_gpu[nyp-2:nyp-1,:,:,2:3] - up_gpu[nyp-1:nyp-0,:,:,2:3])/dy_gpu[nyp-1:nyp,:,:,:]/abs(gamma_gpu[:,:,:,:])
            wf_action_t_gpu[:,:,:] = 1j*gamma_gpu[:,:,:,:]*(up_gpu[nyp-13:nyp-12,:,:,2:3] - up_gpu[nyp-12:nyp-11,:,:,2:3])/dy_gpu[nyp-12:nyp-11,:,:,:]/abs(gamma_gpu[:,:,:,:])
            wf_action_t_gpu[:,0,:] = 0.0
            #print(1j*(up_gpu[nyp-2:nyp-1,:,:,2:3] - up_gpu[nyp-1:nyp-0,:,:,2:3])/dy_gpu[nyp-1:nyp,:,:,:])
            w_action_t[:,:,:] = cp.real(cp.fft.irfft2(wf_action_t_gpu, axes=(1,2))*(nzp*nxp))

            #get constant value ketp 0.15*u(tau) = C*action_rms 
            w_action_b_rms = cp.std(w_action_b[:])
            w_action_t_rms = cp.std(w_action_t[:])


            Constant_b = 0.15/w_action_b_rms
            Constant_t = 0.15/w_action_t_rms

            w_action_b[:] = Constant_b*w_action_b[:] 
            w_action_t[:] = Constant_t*w_action_t[:] 

            vp_bbc_gpu[:] = cp.fft.rfft2(w_action_b[:], axes=(1,2))/(nzp*nxp)
            vp_tbc_gpu[:] = -cp.fft.rfft2(w_action_t[:], axes=(1,2))/(nzp*nxp) 
            '''




            #----------------------------------------------------- Opposition control -----------------------------------------------------#
            #vp_bbc_gpu[:] = -up_gpu[32:33,:,:,1:2]
            #vp_tbc_gpu[:] = -up_gpu[nyp-33:nyp-32,:,:,1:2]

            #----------------------------------------------------- Reinforcement learning control -----------------------------------------------------#
            vp_bbc_gpu[:] = action_output_gpu_b 
            vp_tbc_gpu[:] = -action_output_gpu_t  

            up_gpu[0:1,:,:,1:2] = vp_bbc_gpu[:] #up는 fourier domain에 있다.
            up_gpu[nyp-1:nyp,:,:,1:2] = vp_tbc_gpu[:]

            zero_padding(up_gpu)

            #Get velocity and gradient of velocity (physical space)
            dudxp_gpu = 1j*alpha_gpu*up_gpu[:,:,:,0:3]; dudzp_gpu = 1j*gamma_gpu*up_gpu[:,:,:,0:3]
            u_gpu = cp.real(cp.fft.irfft2(up_gpu, axes=(1,2))*(nzp*nxp))
            dudx_gpu = cp.real(cp.fft.irfft2(dudxp_gpu, axes=(1,2))*(nzp*nxp))
            dudz_gpu = cp.real(cp.fft.irfft2(dudzp_gpu, axes=(1,2))*(nzp*nxp))
            dudy_gpu[1:nyp-1] = 0.5*(u_gpu[1:nyp-1]-u_gpu[0:nyp-2])/(dy_gpu[1:nyp-1]) \
                               +0.5*(u_gpu[2:nyp]-u_gpu[1:nyp-1])/(dy_gpu[2:nyp])
            
            #nonlinear term momentum
            H1p_gpu[:] = H2p_gpu[:]
            H2_gpu = u_gpu[1:nyp-1,:,:,0:1]*dudx_gpu[1:nyp-1] + u_gpu[1:nyp-1,:,:,1:2]*dudy_gpu[1:nyp-1] + u_gpu[1:nyp-1,:,:,2:3]*dudz_gpu[1:nyp-1] 
        
            #H2_gpu[:,:,:,0:1] += -1 #<dp/dx>=-1 --> 이게 뭐지?
            H2p_gpu[:] = cp.fft.rfft2(H2_gpu, axes=(1,2))/(nxp*nzp)
            
            if t_real == 0: H1p_gpu[:] = H2p_gpu[:]
        
            #velocity TDMA
            d_gpu = ((1.0-dt/(2*Re)*alpha_gpu**2-dt/(2*Re)*gamma_gpu**2)*up_gpu[1:nyp-1]
             +dt/(2*Re)*(+(up_gpu[2:nyp]-up_gpu[1:nyp-1])/dy_gpu[2:nyp]
                         -(up_gpu[1:nyp-1]-up_gpu[0:nyp-2])/dy_gpu[1:nyp-1])
                         /(0.5*dy_gpu[1:nyp-1]+0.5*dy_gpu[2:nyp])
             +dt*(-1.5*H2p_gpu[0:nyp-2]+0.5*H1p_gpu[0:nyp-2])) 

            d_gpu[0:1,:,:,1:2] += - a_gpu[0:1] * vp_bbc_gpu[:]
            d_gpu[nyp-3:nyp-2,:,:,1:2] += - c_gpu[nyp-3:nyp-2] * vp_tbc_gpu[:]

            up_mid_gpu[1:nyp-1,:,:,:] = TDMA_gpu(a_gpu,b_gpu,c_gpu,d_gpu,wtdma_gpu,gtdma_gpu,ptdma_gpu,nyp-2) 
            up_mid_gpu[0:1] = 0.0 #여기 바뀌어야 함
            up_mid_gpu[nyp-1:nyp] = 0.0 #여기도 바뀌어야 함
            up_mid_gpu[0:1,:,:,1:2] = vp_bbc_gpu[:]
            up_mid_gpu[nyp-1:nyp,:,:,1:2] = vp_tbc_gpu[:]
        
            #pressure TDMA
            dp_gpu[0:nyp-1,:,:] = 1.0/dt*(+1j*alpha_gpu*(0.5*up_mid_gpu[0:nyp-1,:,:,0:1]+0.5*up_mid_gpu[1:nyp,:,:,0:1]) 
                                          +1j*gamma_gpu*(0.5*up_mid_gpu[0:nyp-1,:,:,2:3]+0.5*up_mid_gpu[1:nyp,:,:,2:3])
                                          +(up_mid_gpu[1:nyp,:,:,1:2]-up_mid_gpu[0:nyp-1,:,:,1:2])/dy_gpu[1:nyp])
        

            pp_gpu[1:nyp,:,:,:] = TDMA_gpu(ap_gpu,bp_gpu,cp_gpu,dp_gpu,wtdma_p_gpu,gtdma_p_gpu,ptdma_p_gpu,nyp-1)
        
            pp_gpu[0:1,:,:] = pp_gpu[1:2,:,:]
            pp_gpu[nyp:nyp+1,:,:] = pp_gpu[nyp-1:nyp,:,:]
          
            #get pressure gradient
            dpdxp_gpu = 1j*alpha_gpu*(0.5*pp_gpu[0:nyp]+0.5*pp_gpu[1:nyp+1])
            dpdzp_gpu = 1j*gamma_gpu*(0.5*pp_gpu[0:nyp]+0.5*pp_gpu[1:nyp+1])                     
            dpdyp_gpu = (pp_gpu[1:nyp+1]-pp_gpu[0:nyp])/(0.5*dy_gpu[1:nyp+1]+0.5*dy_gpu[0:nyp])
        
            ##mean pressure gradient (<dpdx> = -1) --> nonlinear term
            #dpdxp_gpu[:,0:1,0:1] = dpdxp_gpu[:,0:1,0:1] - 1
        
            # pressure correction (pressure term에 의한 update 부분)
            up_gpu[1:nyp-1,:,:,0:1] = up_mid_gpu[1:nyp-1,:,:,0:1] - dt * dpdxp_gpu[1:nyp-1]
            up_gpu[1:nyp-1,:,:,1:2] = up_mid_gpu[1:nyp-1,:,:,1:2] - dt * dpdyp_gpu[1:nyp-1]
            up_gpu[1:nyp-1,:,:,2:3] = up_mid_gpu[1:nyp-1,:,:,2:3] - dt * dpdzp_gpu[1:nyp-1]

            zero_padding(up_gpu)

            #Constant flow 유량 일정하게 다시 바꿔주는 부분
            up_storage_gpu[:,:,:,:] = cp.real(np.fft.irfft2(up_gpu, axes=(1,2))*(nxp*nzp))            

            mean_u_gpu = cp.mean(up_storage_gpu[:,:,:,0], axis=(1,2))
            bulk_v_gpu = cp.sum(mean_u_gpu[:] * (0.5*dy_gpu[0:nyp,0,0,0] + 0.5*dy_gpu[1:nyp+1,0,0,0])) 
            bulk_v_gpu = bulk_v_gpu/(float(yl))  
            dpdx_mean_gpu = (bulk_v_gpu - bulk_v_init_gpu) / dt

            print("Test check of dpdx")
            print(dpdx_mean_gpu)

            up_gpu[1:nyp-1,0:1,0:1,0:1] = up_gpu[1:nyp-1,0:1,0:1,0:1] - dt * cp.reshape(dpdx_mean_gpu, [1,1,1,1])

            div_gpu = cp.mean((cp.fft.irfft2(+1j*alpha_gpu*(0.5*up_gpu[0:nyp-1,:,:,0:1]+0.5*up_gpu[1:nyp,:,:,0:1]) 
                                                       +1j*gamma_gpu*(0.5*up_gpu[0:nyp-1,:,:,2:3]+0.5*up_gpu[1:nyp,:,:,2:3])
                                                       +(up_gpu[1:nyp,:,:,1:2]-up_gpu[0:nyp-1,:,:,1:2])/dy_gpu[1:nyp], axes=(1,2))*(nxp*nzp))**2)**0.5       
            div = cp.asnumpy(div_gpu)/Re



            ########################################## 한 step의 속도장 계산은 여기서 끝난다 ############################################
            
            #Shear stress (wall velocity gradient)
            mean_shear_b_gpu = cp.real(cp.mean((up_gpu[1:2,0:1,0:1,0:1]-up_gpu[0:1,0:1,0:1,0:1])/dy_gpu[1:2]))
            mean_shear_t_gpu = cp.real(cp.mean((up_gpu[nyp-2:nyp-1,0:1,0:1,0:1]-up_gpu[nyp-1:nyp,0:1,0:1,0:1])/dy_gpu[nyp-1:nyp]))

            
            #Fourier domain to Real domain (to obtain statsitcs)
            up_storage_gpu[:,:,:,:] = cp.real(np.fft.irfft2(up_gpu, axes=(1,2))*(nxp*nzp))
            
            # Vorticity 구해주는 부분 #
            
            #vorticity (x-direction) d(v_z)/dy - d(v_y)/dz
            vorticity_p_gpu[:,:,:,0:1] = (up_gpu[1:nyp,:,:,2:3] - up_gpu[0:nyp-1,:,:,2:3])/dy_gpu[0:nyp-1,:, :,:] \
                            - (1j*gamma_gpu*(up_gpu[0:nyp-1,:,:,1:2] + up_gpu[1:nyp,:,:,1:2])/2)

            #vorticity (y-direction) d(v_x)/dz - d(v_z)/dx
            vorticity_p_gpu[:,:,:,1:2] = 1j*gamma_gpu*((up_gpu[0:nyp-1,:,:,0:1] + up_gpu[1:nyp,:,:,0:1])/2) - 1j*alpha_gpu*((up_gpu[0:nyp-1,:,:,2:3] + up_gpu[1:nyp,:,:,2:3])/2)

            #vorticity (z-direction) d(v_y)/dx - d(v_x)/dy
            vorticity_p_gpu[:,:,:,2:3] = 1j*alpha_gpu*((up_gpu[0:nyp-1,:,:,1:2] + up_gpu[1:nyp,:,:,1:2])/2) \
                        - (up_gpu[1:nyp,:,:,0:1] - up_gpu[0:nyp-1,:,:,0:1])/dy_gpu[0:nyp-1,:, :,:]

            vorticity_gpu[:,:,:,:] = cp.real(cp.fft.irfft2(vorticity_p_gpu, axes=(1,2))*(nxp*nzp))  



            #average & std 구해주는 부분#
            mean_u_gpu = cp.mean(up_storage_gpu[:,:,:,0], axis=(1,2)) #mean velocity profile
            rms_u_gpu = cp.std(up_storage_gpu[:,:,:,:], axis=(1,2)) #RMS (x, z에 대해서는 고정해놓고 y에 대해서만)
            rms_vorticity_gpu = cp.std(vorticity_gpu[:,:,:,:], axis=(1,2))   

            #---------------------------------- Two-point correlation - streamwise ------------------------------------------#
            #y+ ~ 10 data
            if t > starting_avg:
                u_fluc_gpu[:] = up_storage_gpu[32,:,:,0] - cp.mean(up_storage_gpu[32,:,:,0])
                v_fluc_gpu[:] = up_storage_gpu[32,:,:,1] - cp.mean(up_storage_gpu[32,:,:,1])
                w_fluc_gpu[:] = up_storage_gpu[32,:,:,2] - cp.mean(up_storage_gpu[32,:,:,2])

                u_fluc_stack_gpu[f_t:f_t+1,:,:] = u_fluc_gpu[:]
                v_fluc_stack_gpu[f_t:f_t+1,:,:] = v_fluc_gpu[:]
                w_fluc_stack_gpu[f_t:f_t+1,:,:] = w_fluc_gpu[:]

                if f_t >= fluc_nums: f_t = 0
    
                f_t +=1


                if t % time_avg_snap ==0:

                    for x in range(nxp):
                        R_11_stream_gpu[x] = cp.mean((u_fluc_stack_gpu[:,:,0] - cp.mean(u_fluc_stack_gpu[:,:,0]))*(u_fluc_stack_gpu[:,:,x]-cp.mean(u_fluc_stack_gpu[:,:,x])))\
                        /(cp.std(u_fluc_stack_gpu[:,:,0])*cp.std(u_fluc_stack_gpu[:,:,x]))
                        R_22_stream_gpu[x] = cp.mean((v_fluc_stack_gpu[:,:,0] - cp.mean(v_fluc_stack_gpu[:,:,0]))*(v_fluc_stack_gpu[:,:,x]-cp.mean(v_fluc_stack_gpu[:,:,x])))\
                        /(cp.std(v_fluc_stack_gpu[:,:,0])*cp.std(v_fluc_stack_gpu[:,:,x]))
                        R_33_stream_gpu[x] = cp.mean((w_fluc_stack_gpu[:,:,0] - cp.mean(w_fluc_stack_gpu[:,:,0]))*(w_fluc_stack_gpu[:,:,x]-cp.mean(w_fluc_stack_gpu[:,:,x])))\
                        /(cp.std(w_fluc_stack_gpu[:,:,0])*cp.std(w_fluc_stack_gpu[:,:,x]))

                    Two_correlation = open(Test_time_avg + "/two_point correlation time{}.plt".format(t), 'a', encoding='utf-8', newline='') 

                    Two_correlation.write('VARIABLES = "x", "Correlation_f", "Correlation_g", "Correlation_w"\n')

                    for i in range(nxp):
                        Two_correlation.write('%.5f %.5f %.5f %.5f\n'%(i*dx, R_11_stream_gpu[i], R_22_stream_gpu[i], R_33_stream_gpu[i]))

                    Two_correlation.close()


            ################################### time average statisitics ##################################
            
            if t>starting_avg:
                u_t_avg_gpu[:] = mean_u_gpu[:]*dt + u_t_avg_gpu[:]
                rms_t_avg_gpu[:,0] = rms_u_gpu[:,0]*dt + rms_t_avg_gpu[:,0]  
                rms_t_avg_gpu[:,1] = rms_u_gpu[:,1]*dt + rms_t_avg_gpu[:,1]
                rms_t_avg_gpu[:,2] = rms_u_gpu[:,2]*dt + rms_t_avg_gpu[:,2]
                vorticity_t_avg_gpu[:,0] = rms_vorticity_gpu[:,0]*dt + vorticity_t_avg_gpu[:,0]
                vorticity_t_avg_gpu[:,1] = rms_vorticity_gpu[:,1]*dt + vorticity_t_avg_gpu[:,1]
                vorticity_t_avg_gpu[:,2] = rms_vorticity_gpu[:,2]*dt + vorticity_t_avg_gpu[:,2]
                Re_t_avg_gpu[:] = (mean_shear_b_gpu + mean_shear_t_gpu)/2*dt + Re_t_avg_gpu[:]


            if t>starting_avg and t % time_avg_snap == 0:

                t_avg = t-starting_avg

                rms_t_avg = cp.asnumpy(rms_t_avg_gpu) 
                u_t_avg = cp.asnumpy(u_t_avg_gpu)

                f_mean = open(Test_time_avg + '/time_mean velocity time_step {}.plt' .format(t), 'a', encoding='utf-8', newline='')
                f_rms_t = open(Test_time_avg + '/time_vel_rms time_step {}.plt' .format(t), 'a', encoding='utf-8', newline='')
                f_vor_rms = open(Test_time_avg + '/time_vort_rms time{}.plt'.format(t), 'a', encoding='utf-8', newline='')
                f_Re_t = open(Test_time_avg + '/time_Re_tau time{}.plt'.format(t), 'a', encoding='utf-8', newline='')

                f_mean.write('VARIABLES = "y", "u_mean" \n')	
                f_rms_t.write('VARIABLES = "y", "u(y)_rms", "v(y)_rms", "w(y)_rms" \n')
                f_vor_rms.write('VARIABLES = "y", "vor(x)_rms", "vor(y)_rms", "vor(z)_rms" \n')
                f_Re_t.write('VARIABLES = "t", "Re_actual_avg"\n')

                for j in range(nyp):
                    f_mean.write('%.5f %.5f \n' %(y[j], u_t_avg[j]/(dt*t_avg)))    
                    f_rms_t.write('%.5f  %.5f  %.5f  %.5f \n' %(y[j], rms_t_avg[j,0]/(dt*t_avg), rms_t_avg[j,1]/(dt*t_avg), rms_t_avg[j,2]/(dt*t_avg)))

                #Vorticity RMS 
                for j in range(nyp-1):
                    f_vor_rms.write('%.5f  %.5f  %.5f  %.5f \n' %((y[j]+y[j+1])/2, vorticity_t_avg[j,0]/(dt*t_avg), vorticity_t_avg[j,1]/(dt*t_avg), vorticity_t_avg[j,2]/(dt*t_avg)))
                
                f_Re_t.write('%.5f %.5f \n'%(t, Re_t_avg_gpu/(dt*t_avg)))

                f_mean.close()
                f_rms_t.close()
                f_vor_rms.close()
                f_Re_t.close()

            ##############################################################################################
        
        
            #Bulk Velocity 유량 바꿔주는 부분#
            bulk_v_gpu = cp.sum(mean_u_gpu[:] * (0.5*dy_gpu[0:nyp,0,0,0] + 0.5*dy_gpu[1:nyp+1,0,0,0]))
            bulk_v_gpu = bulk_v_gpu/(float(yl))          
            bulk_v_return = cp.asnumpy(bulk_v_gpu) 

            #Drag Coefficient 구하는 부분#
            friction_coff_b_gpu = mean_shear_b_gpu*2.0/(180*bulk_v_gpu**2)
            friction_coff_t_gpu = mean_shear_t_gpu*2.0/(180*bulk_v_gpu**2)

            if t_real == 0: #(t_real은 진짜 시간이다)
                initial_coeff_gpu  =(friction_coff_b_gpu + friction_coff_t_gpu)/2


            #(du/dy)
            shear_b_next_du_gpu = (up_storage_gpu[4+wall:5+wall,:,:,0:1] - up_storage_gpu[3+wall:4+wall,:,:,0:1])/dy_gpu[4+wall:5+wall]
            shear_t_next_du_gpu = (up_storage_gpu[nyp-4-wall:nyp-3-wall,:,:,0:1] - up_storage_gpu[nyp-5-wall:nyp-4-wall,:,:,0:1])/dy_gpu[nyp-4-wall:nyp-3-wall]

            #(dw/dy)
            shear_b_next_dw_gpu = (up_storage_gpu[4+wall:5+wall,:,:,2:3] -up_storage_gpu[3+wall:4+wall,:,:,2:3])/dy_gpu[4+wall:5+wall]
            shear_t_next_dw_gpu = (up_storage_gpu[nyp-4-wall:nyp-3-wall,:,:,2:3] - up_storage_gpu[nyp-5-wall:nyp-4-wall,:,:,2:3])/dy_gpu[nyp-4-wall:nyp-3-wall]            

            #(pressure)
            shear_b_next_du= cp.asnumpy(shear_b_next_du_gpu); shear_t_next_du= cp.asnumpy(shear_t_next_du_gpu)
            shear_b_next_dw= cp.asnumpy(shear_b_next_dw_gpu); shear_t_next_dw= cp.asnumpy(shear_t_next_dw_gpu)
            

            #(du/dy - wall)
            shear_wall_dudy = (up_storage_gpu[1:2,:,:,0:1] - up_storage_gpu[0:1,:,:,0:1])/(Re*dy_gpu[1:2])

            #(dw/dy - wall)
            shear_wall_dwdy = (up_storage_gpu[1:2,:,:,2:3] - up_storage_gpu[0:1,:,:,2:3])/(Re*dy_gpu[1:2])

            next_state_stock = (shear_b_next_du[0, :, :, 0]/Re, shear_t_next_du[0, :, :, 0]/Re, shear_b_next_dw[0, :, :, 0]/Re, shear_t_next_dw[0, :, :, 0]/Re)
            #[nzp, nxp] 필요한 정보만 따로 빼내어 주도록 하자 
            next_state_stock = list(next_state_stock)
            self.next_state_b[:,:,0] = next_state_stock[0]#; self.next_state_b[:,:,1] = next_state_stock[2]
            self.next_state_t[:,:,0] = next_state_stock[1]#; self.next_state_t[:,:,1] = next_state_stock[3]

            self.state_interval_b = np.reshape(self.next_state_b, [1, self.state_shape[0]*self.state_shape[1]*self.state_shape[2]])
            self.state_interval_t = np.reshape(self.next_state_t, [1, self.state_shape[0]*self.state_shape[1]*self.state_shape[2]])
            #Cpu to GPU to record one time step
           
            #통계량 및 vorticty와 같은 field의 특징까지 한 step 구할때까지의 시간측정 및 기록
            duration = timer() - start

            #Monitor scatter plot
            if t_real%scatter_freq ==0:
                monitor.Scatter_plot(up_gpu, next_state_stock, vp_bbc_gpu, vp_tbc_gpu, shear_wall_dudy, shear_wall_dwdy, t_real, episode_t, nxp, nyp, nzp, dx, dy, dz, y, Re)


            #----------------------------- every time step recording -----------------------------#
            #bulk velocity recording every time_step each episode
            monitor.bulk_record(bulk_v_gpu, t_real, episode_t)
            
            #Drag reduction (%) recording every time_step each episode
            #monitor.reduction_percentage(friction_coff_b_gpu, friction_coff_t_gpu, initial_coeff_gpu, t_real, state_step, episode_t)
            
            #Drag reduction (%) recording every time_step each episode
            monitor.reduction_percentage_p(t_real, dpdx_mean_gpu, dpdx_mean_init_gpu, state_step, episode_t)

            #Simulation information (drag coefficent, Operation speed, shear stress etc) every time step each episode
            monitor.simulation_info(t_real, dt, duration, mean_shear_b_gpu ,mean_shear_t_gpu, friction_coff_b_gpu, friction_coff_t_gpu, episode_t)

            #Check of Divergence free condition
            monitor.Divergence(t_real, div, episode_t)

            #Check of CFL num
            monitor.CFL_record(CFL_num_gpu, dt, t_real, episode_t)

            #Recording pressure gradient
            monitor.pressure_gradient(t_real, dpdx_mean_gpu, episode_t)

            #RMS_ Record
            monitor.RMS_record(vp_bbc_gpu, t_real, episode_t, nxp, nzp)


            #if t_real % 5000 ==0:
                #monitor.save_3D_field(up_gpu, episode_t, t_real,nxp, nyp, nzp,)

            #-------------------------------------------------------------------------------------#
            
            #----------------------------- take video periodically -------------------------------#
            monitor.time_video(up_gpu, next_state_stock, vp_bbc_gpu, vp_tbc_gpu, t_real, episode_t, nxp, nyp, nzp, dx, dy, dz, y)
            #-------------------------------------------------------------------------------------#
            
            #----------------------------- take real_time snapshot -------------------------------#
            if t_real % snap_shot_freq_t == 0:
                monitor.snap_shot(up_gpu, next_state_stock, shear_wall_dudy, shear_wall_dwdy, rms_vorticity_gpu, vorticity_gpu, vp_bbc_gpu, vp_tbc_gpu, t_real, episode_t, nxp, nyp, nzp, dx, dy, dz, y)
            #-------------------------------------------------------------------------------------#
            
            t_real += 1
            bulk_v_gpu = 0.0 #Initialization of bulk velocity (state에 대해서 뽑을거면 이거 자리를 좀 바꿔줘야함)
            