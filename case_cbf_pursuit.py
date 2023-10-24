from casadi import *
import numpy as np
import time
from task_schedule import *

###########################################################################    
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!ONLINE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
###########################################################################
#from pycrazyswarm import *
###########################################################################    
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!ONLINE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
###########################################################################
tic_total = time.time()

dummy=1 # for casadi jacobian (output has no effect on the result)
# Define mission parameters
umax=.6 # Max vel. of the system
# RoI coordinates x1 y1 rad1; x2 y2 rad2
roi=.1*np.array([[-5,10,3],[10, 10, 3]]); 
n_tar = np.size(roi,0) # of targets
roi_disj = np.copy(roi) # Create alternative RoI's (to be modified)     
alt_inds=np.zeros(n_tar) # Tasks with an alternative will be nonzero only
rois = [roi,roi_disj] # Create alternative full-RoI lists
# The map listing originals to alternative task indices (itself and its alternatives,i.e. max alt. number is the #columns)
disj_map = np.array([np.arange(0,n_tar)]) 
#this will be reached via disj_map[:,alt_id]

roi_all = np.copy(roi) # Add alternative targets to original targets
n_tar_all = np.size(roi_all,0) # of all targets


t_windows=[[[0,13],[0,2]],[[0,13],[0,2]]] # STL time windows

subformula_types = np.array([3,3]) # 1: F, 2: G, 3: FG, 4: GF | Formula Types

v_tar_max = .1*np.array([.8,.8]) # <- FAST TARGETS NEGATIVE CBF | # SLOW TARGETS (GUANTEE) .1*np.array([.8,.8])
v_tar_max_disj = np.copy(v_tar_max)  # Assume that alternative targets have the same speed bounds
v_tar_max_all = np.copy(v_tar_max) # Maximum vel in any direct. for each target + alternative

xold=np.array([-2,0]) # Initial pos. (x,y) of our system

chain, rem_time, rem_time_seq, gamma, portions, portions0 = task_scheduler(rois,t_windows,subformula_types,xold,umax,v_tar_max)
rem_time = np.array(rem_time)
rem_time_seq = np.array(rem_time_seq)

alpha_b=.1 #.4 -  Sequential CBF tuning parameter
alpha_h=.9 # .9 - Holding CBF -single task with rem_time=1

periods = np.zeros(n_tar)
periodic_tasks=np.where(subformula_types==4)[0]
for i in range(len(periodic_tasks)):
    periods[periodic_tasks[i]] = t_windows[periodic_tasks[i]][1][1]-t_windows[periodic_tasks[i]][1][0]
    
hrz_cand = np.zeros(n_tar)
for i in range(n_tar):
    hrz_cand[i] = sum(item[1] for item in t_windows[i]) # Check the horizons of each subformula
hrz=max(hrz_cand) # Pick the maximum horizon as the mission hrz

u = SX.sym('u',2)  
xc = SX.sym('xc',3) # Circle x,y,rad
xc_pw = SX.sym('xc_pw',3) # Circle x,y,rad
vc = SX.sym('vc'); rc = SX.sym('rc')
vc_pw = SX.sym('vc_pw'); rc_pw = SX.sym('rc_pw')
xp = SX.sym('xp',2) # Point x,y
dist_p_cir = Function('dist_p_cir',[xc,vc,rc,xp],[sqrt((xp[0]-xc[0])**2+(xp[1]-xc[1])**2)-xc[2]+vc*rc],\
                                              ['xc','vc','rc','xp'],['dist'])
dist_cir_cir = Function('dist_cir_cir',[xc,vc,rc,xc_pw,vc_pw,rc_pw],\
                        [sqrt((xc[0]-xc_pw[0])**2+(xc[1]-xc_pw[1])**2)+xc[2]+vc*rc-xc_pw[2]+vc_pw*rc_pw],\
                                              ['xc','vc','rc','xc_pw','vc_pw','rc_pw'],['dist_pw'])

b_minus_dist = Function('b_minus_dist',[vc,rc],[rc*umax-vc*rc])

d_b_minus_dist = b_minus_dist.jacobian()
d_b_dist = dist_p_cir.jacobian()
d_b_pair = dist_cir_cir.jacobian()
                          
x_history = []  
u_history = []
b_history = []                          
                        
x_history.append(xold)

c = 0 # =1 in MATLAB!!!
dropped = 0
counterG = 0
flag_final_hold = 0
periodic_hold = 0
next_tasks = []

elapsed_qp=[]; elapsed_run=[]
tic = time.time()
###########################################################################    
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!ONLINE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
###########################################################################

## Mapping Target Sequence Indices to Crazyflie IDs
#test_id = [0,1] #[0] When single drone is flying to test | 0: system, 1-6: targets | cbf chain order
#seq_to_cf =[2, 6, 4] #[2, 1, 3, 4, 5, 6, 7]  # 0: system, 1-6: targets | 7 cfs total
#z_cfs = [1.1,.4,.4] # Altitudes: system at 1 m, targets at 0.5 m
#positions_all = np.vstack((xold,roi_all[:,:2]))
##print(positions_all)
## execute waypoints
#swarm = Crazyswarm()
#timeHelper = swarm.timeHelper
#allcfs = swarm.allcfs
#
## All take-off!
#allcfs.takeoff(targetHeight=0.6, duration=2.0)
#timeHelper.sleep(2.0)
#
## All get your respective initial positions!
#for id in range(len(seq_to_cf)): #test_id:
#    agent = seq_to_cf[id]
#    #print(agent)
#    pos = list(positions_all[id,:]); pos.append(z_cfs[id])
#    #print(pos)
#    cf = allcfs.crazyfliesById[agent]
#    cf.goTo(pos, 0, 2.1)
#done_tasks = [] # 1 to 6 NOT 0 to 5 (hence +1 in the definitions below)
###########################################################################    
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!ONLINE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
###########################################################################
    

for i in range(int(hrz)):
    tic_run = time.time()
    counterG -= 1
    if flag_final_hold == 1 or dropped == len(chain):
        rb = 1000
        c = len(chain)-1
    else:
        rb = rem_time_seq[gamma]
    
    
    v_targets_all_x = []; v_targets_all_y = []
    for chase in range(len(chain)):
        abs_dist = max(0,dist_p_cir(roi[chain[chase],:],0,0,xold).full())
        if abs_dist>=2:
            v_targets_all_x.append(0)
            v_targets_all_y.append(-v_tar_max[1])
        else:
            vx_target = (roi[chain[chase],0]-xold[0])\
            /abs(roi[chain[chase],0]-xold[0])*v_tar_max[chain[chase]]\
            /sqrt(2)*(1-exp(-abs_dist))

            vy_target = (roi[chain[chase],1]-xold[1])\
            /abs(roi[chain[chase],1]-xold[1])*v_tar_max[chain[chase]]\
            /sqrt(2)*(1-exp(-abs_dist))

            v_targets_all_x.append(vx_target)
            v_targets_all_y.append(vy_target)
    v_targets_all =  [np.array(v_targets_all_x),np.array(v_targets_all_y)]

    
    v_targets = [v_targets_all[0][:n_tar],v_targets_all[1][:n_tar]]
    const = 0 # Value of b!!!
    d_b_pair_cir_sum = 0
    d_b_pair_r_sum = 0
    for j in range(c+1,gamma+1):
        xc_v = roi[chain[j-1],:]; xc_pw_v = roi[chain[j],:]
        vc_v = v_tar_max[chain[j-1]]; vc_pw_v = v_tar_max[chain[j]]
        rc_v = rem_time_seq[j-1]; rc_pw_v = rem_time_seq[j]
        const += dist_cir_cir(xc_v,vc_v,rc_v,xc_pw_v,vc_pw_v,rc_pw_v) 
        if subformula_types[chain[j-1]] == 2: # G
            hold_time = t_windows[chain[j-1]][0][1] - t_windows[chain[j-1]][0][0]
            const += hold_time*v_tar_max[chain[j-1]]/umax+hold_time
        elif subformula_types[chain[j-1]] == 3: # FG
            hold_time = t_windows[chain[j-1]][1][1] - t_windows[chain[j-1]][1][0]
            const += hold_time*v_tar_max[chain[j-1]]/umax+hold_time
            
        d_b_pair_cir_sum += dot(d_b_pair(xc_v,vc_v,rc_v,xc_pw_v,vc_pw_v,rc_pw_v,dummy)[[0,1,5,6]],\
         horzcat(v_targets[0][chain[j-1]],v_targets[1][chain[j-1]],v_targets[0][chain[j]],v_targets[1][chain[j]]))
        d_b_pair_r_sum += sum2(d_b_pair(xc_v,vc_v,rc_v,xc_pw_v,vc_pw_v,rc_pw_v,dummy)[[4,9]])
    xp_v = xold
    xc_v = roi[chain[c],:]
    vc_v = v_tar_max[chain[c]]
    rc_v = rb
    v_target = [v_targets[0][chain[c]],v_targets[1][chain[c]]]
    
    d_b_x = 0-d_b_dist(xc_v,vc_v,rc_v,xp_v,dummy)[-2:] # wrt sys_x, sys_y 
    d_b_rb = d_b_minus_dist(vc_v,rc_v,dummy)[1]-d_b_dist(xc_v,vc_v,rc_v,xp_v,dummy)[4]  # wrt rc
    d_b_cir = 0-d_b_dist(xc_v,vc_v,rc_v,xp_v,dummy)[:2]  # wrt cir_x,cir_y 
    
    alpha = alpha_b
    A_barrier = -d_b_x
    B_barrier = -1*d_b_rb + dot(d_b_cir,np.array([v_target]))\
                          +alpha*(b_minus_dist(vc_v,rc_v)-dist_p_cir(xc_v,vc_v,rc_v,xp_v))\
                          -alpha*const-d_b_pair_cir_sum+d_b_pair_r_sum
    b_history.append(b_minus_dist(vc_v,rc_v)-dist_p_cir(xc_v,vc_v,rc_v,xp_v)-const)
    
    if subformula_types[chain[max(0,c-1)]] == 4 and dist_p_cir(roi[chain[max(0,c-1)],:],0,0,xold).full()>=0.01:
        periodic_hold = 0
        if any(next_tasks):
            next_per_tasks_idx = np.where(next_tasks)[0]
            rem_time[c+next_per_tasks_idx[0]] = periods[chain[c-1]]-1
            rem_time_seq[c+next_per_tasks_idx[0]] = min(rem_time[next_per_tasks_idx[0]+c:])
            portions[c+next_per_tasks_idx[0]]=portions0[c+next_per_tasks_idx[0]]
            next_tasks=[]
    if counterG>0 or periodic_hold ==1 or flag_final_hold==1:
        A_barrier = vertcat(A_barrier,A_barrierG)
        B_barrier = vertcat(B_barrier,B_barrierG)
        
    ###########################################################################    
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!QP PROBLEM!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
    ###########################################################################
    tic_qp = time.time()
    qp = {'x':u, 'f':dot(u,u), 'g':B_barrier-mtimes(A_barrier,u)}
    opts = {'printLevel':'low'}
    S = qpsol('S', 'qpoases', qp, opts)
    opt = S(lbg=0)
    u_opt = opt['x']
    elapsed_qp.append(time.time()-tic_qp)
    ###########################################################################    
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!QP PROBLEM!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
    ###########################################################################

    xold=xold+u_opt
    u_history.append(u_opt)
    x_history.append(xold)

    roi[:,0] = roi[:,0]+v_targets[0]; roi[:,1] = roi[:,1]+v_targets[1]
    roi_all[:,0] = roi_all[:,0]+v_targets_all[0]; roi_all[:,1] = roi_all[:,1]+v_targets_all[1]
    
    ###########################################################################    
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!ONLINE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
    ###########################################################################
    
#    timeHelper.sleep(1.0) # Each loop supposed to take 1 sec
#    positions_all = np.vstack((xold.full().T,roi_all[:,:2]))
#
#    # All get your respective initial positions!
#    for id in range(len(seq_to_cf)): #test_id:
#
#        agent = seq_to_cf[id]
#        #print(agent)
#        pos = list(positions_all[id,:]); pos.append(z_cfs[id])
#        #print(pos)
#        cf = allcfs.crazyfliesById[agent]
#
#	if id in done_tasks:
#	    #land
#	    cf.land(targetHeight=0.06, duration=1.0)
#	else:
#	    cf.goTo(pos, 0, 1.0) # 1 is the duration, time step
#    # Holding completed can land now (previous in the chain)
#    if counterG == 0 and alt_inds[chain[c-1]]>0: # Land alternative targets together when one of them is done
#	done_tasks.extend(list(disj_map[:,chain[c-1]][disj_map[:,chain[c-1]]!=0]+1))
#
#    elif counterG == 0: # No disjunction
#	done_tasks.append(chain[c-1]+1)


    ###########################################################################    
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!ONLINE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
    ###########################################################################

    rem_time -= 1
    rem_time_seq -= 1
    
    if dist_p_cir(roi[chain[c],:],0,0,xold).full()<=0.01:
        dropped += 1
        if subformula_types[chain[c]] != 1: # NEED TO BE HELD
            if dropped == len(chain):
                flag_final_hold = 1
                
            rb = 1; rc_v = rb
            xp_v = xold
            xc_v = roi[chain[c],:]
            #vc_v is the same
            
            d_b_x = 0-d_b_dist(xc_v,vc_v,rc_v,xp_v,dummy)[-2:] # wrt sys_x, sys_y | nothing comes from minus_dist
            d_b_rb = d_b_minus_dist(vc_v,rc_v,dummy)[1]-d_b_dist(xc_v,vc_v,rc_v,xp_v,dummy)[4]  # wrt rc
            d_b_cir = 0-d_b_dist(xc_v,vc_v,rc_v,xp_v,dummy)[:2]  # wrt cir_x,cir_y | nothing comes from minus_dist
            
            alpha = alpha_h
            A_barrierG = -d_b_x
            B_barrierG = -1*d_b_rb + dot(d_b_cir,np.array([v_target]))\
                          +alpha*(b_minus_dist(vc_v,rc_v)-dist_p_cir(xc_v,vc_v,rc_v,xp_v))
            
            if subformula_types[chain[c]] == 4:
                periodic_hold = 1
                next_tasks = [p_idx==chain[c] for p_idx in chain[c+1:]]
            elif subformula_types[chain[c]] == 2:
                counterG = t_windows[chain[c]][0][1]-i+1
            elif subformula_types[chain[c]] == 3:
                counterG = max((t_windows[chain[c]][1][1]-t_windows[chain[c]][1][0]),\
                           (t_windows[chain[c]][0][0]-t_windows[chain[c]][1][1]-i+1))
    ###########################################################################    
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!ONLINE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
    ########################################################################### 
#        else: # Eventually tasks are hit and done
# 	    if alt_inds[chain[c]]>0: # Land alternative targets together when one of them is done
#	        done_tasks.extend(list(disj_map[:,chain[c]][disj_map[:,chain[c]]!=0]+1))
#
#            else: # No disjunction
#		done_tasks.append(chain[c]+1)
#

    ###########################################################################    
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!ONLINE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
    ###########################################################################    
        if c==gamma: 
            try:
                newgamma = np.argmin(portions[gamma+1:]) 
            except:
                newgamma = 0
            gamma=min(gamma+newgamma+1,len(chain)-1)
        c += 1; # Switch to next formula
            
        ##### NEXT, CHECK IF DISJUNCTION
        if alt_inds[chain[min(c,len(chain)-1)]]>0: # IT IS A TASK WITH ALTERNATIVE
            b_val = []
            for alts in range(np.count_nonzero(disj_map[:,chain[4]])): # Add First two elements in the sequence to rest (only these variable if disjunction)
                alt_task = disj_map[alts,chain[c]]
                v_target = [v_targets_all[0][alt_task],v_targets_all[1][alt_task]]
                xp_v = xold
                xc_v = roi_all[alt_task,:]; xc_pw_v = roi_all[chain[c+1],:]
                vc_v = v_tar_max[chain[c]]; vc_pw_v = v_tar_max[chain[c+1]] # Same max vel for all
                rc_v = rem_time_seq[c]; rc_pw_v = rem_time_seq[c+1] # No temp. disj., same rem time
                const = dist_cir_cir(xc_v,vc_v,rc_v,xc_pw_v,vc_pw_v,rc_pw_v) 
                if subformula_types[chain[c]] == 2: # G
                    hold_time = t_windows[chain[c]][0][1] - t_windows[chain[c]][0][0]
                    const += hold_time*v_tar_max[chain[c]]/umax+hold_time
                elif subformula_types[chain[j-1]] == 3: # FG
                    hold_time = t_windows[chain[c]][1][1] - t_windows[chain[c]][1][0]
                    const += hold_time*v_tar_max[chain[c]]/umax+hold_time
        
                b_val.append(b_minus_dist(vc_v,rc_v)-dist_p_cir(xc_v,vc_v,rc_v,xp_v)-const)
            
            #UPDATE CHAIN HERE IF NEEDED
            best_alt = np.argmax(b_val)
            roi[chain[c],:] = roi_all[disj_map[best_alt,chain[c]],:]
            coeffs_all[chain[c],:] = coeffs_all[disj_map[best_alt,chain[c]],:]
            v_targets[0][chain[c]] = v_targets_all[0][disj_map[best_alt,chain[c]]]
            v_targets[1][chain[c]] = v_targets_all[1][disj_map[best_alt,chain[c]]]
    elapsed_run.append(time.time()-tic_run)
###########################################################################    
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!ONLINE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
###########################################################################
#
## land
#allcfs.land(targetHeight=0.06, duration=2.0)
#timeHelper.sleep(2.0)

###########################################################################    
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!ONLINE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
###########################################################################
elapsed = time.time() - tic
print({'Average QP Soln Time':sum(elapsed_qp)/hrz})
print({'Average Online Runtime':sum(elapsed_run)/hrz})
print({'Total Online Runtime':sum(elapsed_run)})

print({'Total Execution Time':time.time()-tic_total})
