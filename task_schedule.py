import numpy as np
from casadi import *
import itertools
import time

def task_scheduler(rois,t_windows,subformula_types,x0,umax,v_tar_max):
    feasible_chains = []
    feasible_portions = []
    feasible_rem_time = []
    minimum_portions = []
    n_alt = len(rois)
    for alt in range(n_alt): # Alternatives
        roi_temp = rois[alt]
        chain,portions,rem_time,min_portions = sequence_constructer(t_windows,subformula_types,x0,roi_temp,umax,v_tar_max)
        feasible_chains.append(chain)
        feasible_portions.append(portions)
        feasible_rem_time.append(rem_time)
        minimum_portions.append(min_portions)
    best_alt = np.argmax(minimum_portions)
    roi=rois[best_alt]
    chain = feasible_chains[best_alt]
    portions = feasible_portions[best_alt]
    portions0 = np.copy(portions)
    rem_time = feasible_rem_time[best_alt]
    first_appear = np.unique(chain,return_index=True)[1] # unique tasks [0] and order of their first appearance [1]
    try:
        portions[np.setdiff1d(np.array([i for i in range(len(portions))]),first_appear)]=1000; # Periodic tasks doesn't play a role in tightest portion calc.
        rem_time[np.setdiff1d(np.array([i for i in range(len(portions))]),first_appear)]=1000; # Periodic tasks doesn't play a role in tightest portion calc.
    except:
        pass
    gamma=np.argmin(portions); # Find the tightest portion
    rem_time_seq = [min(rem_time[i_r:]) for i_r in range(len(chain))] # Realistic remaining times (a target can move at most rr step)  
    return chain, rem_time, rem_time_seq, gamma, portions, portions0

def sequence_constructer(t_windows,types,x0,roi,umax,v_maxs):
    n_tar = len(t_windows) # of subformulas
    r = np.zeros([n_tar])
    rr = np.zeros([n_tar])
    hrz_cand = np.zeros(n_tar)
    for i in range(n_tar):
        hrz_cand[i] = sum(item[1] for item in t_windows[i])# Check the horizons of each subformula
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
    for i in range(n_tar):
        if types[i]==0: # already achieved
            r[i] = 100000
        elif types[i]==1: #F_[a,b]
            r[i] = t_windows[i][0][1] # b
        elif types[i]==2: #G_[a,b]
            r[i] = t_windows[i][0][0] # a
        elif types[i]==3: #F_[a,b]G_[c,d]
            r[i] = t_windows[i][0][1]+t_windows[i][1][0] # b+c
        elif types[i]==4: #G_[a,b]F_[c,d]
            r[i] = t_windows[i][0][0]+t_windows[i][1][1] # a+d
    feasible_orders = []
    laxity_vals = []
    periodic_tasks = np.where(types == 4)[0] # starts from 0 (if any!)
    order_set = interval_clustering(t_windows,periodic_tasks)  
    for order in order_set:
        r_temp=np.copy(r)
        spent_time=0
        flag=0
        # Calculate the realistic remaining times based on the next taks
        i_r = 0
        for seq in order:
            rr[seq] = min([r[i] for i in order[i_r:]]) # These are with real subtask indices
            # Not the order in the sequence
            i_r += 1
        # Calculate the worst-case distances among the sequence
        d=10000*np.ones([len(order),len(order)]) # BETTER TO SET HIGH
        i_seq = 0
        for seq in order:
            if types[seq]==0: # already achieved
                for j in order:
                    d[seq,j] = 0
            elif types[seq]==1: # F_[a,b]
                if i_seq==0:
                    d[seq,seq] = dist_p_cir(roi[seq,:],v_maxs[seq],rr[seq],x0)/umax
                if i_seq==len(order)-1:
                    if any(periodic_tasks):
                        for j in np.setdiff1d(periodic_tasks,seq): # periodic tasks except the task itself
                            d[seq,j] = dist_cir_cir(roi[seq,:],v_maxs[seq],rr[seq],\
                                                roi[j,:],v_maxs[j],rr[j])/umax
                else:
                    for j in np.union1d(order[i_seq+1],np.setdiff1d(periodic_tasks,seq)): # next target and periodic tasks except the task itself
                        d[seq,j] = dist_cir_cir(roi[seq,:],v_maxs[seq],rr[seq],\
                                                roi[j,:],v_maxs[j],rr[j])/umax
            elif types[seq]==2: #G_[a,b]
                hold_time = t_windows[seq][0][1]-t_windows[seq][0][0]
                if i_seq==0:
                    d[seq,seq] = dist_p_cir(roi[seq,:],v_maxs[seq],rr[seq],x0)/umax
                if i_seq==len(order)-1:
                    if any(periodic_tasks):
                        for j in np.setdiff1d(periodic_tasks,seq): # periodic tasks except the task itself
                            d[seq,j] = (dist_cir_cir(roi[seq,:],v_maxs[seq],rr[seq],\
                                                roi[j,:],v_maxs[j],rr[j])+hold_time*v_maxs[seq])/umax\
                                                +hold_time
                else:
                    for j in np.union1d(order[i_seq+1],np.setdiff1d(periodic_tasks,seq)): # next target and periodic tasks except the task itself
                        d[seq,j] = (dist_cir_cir(roi[seq,:],v_maxs[seq],rr[seq],\
                                                roi[j,:],v_maxs[j],rr[j])+hold_time*v_maxs[seq])/umax\
                                                +hold_time

            elif types[seq]==3: #F_[a,b]G_[c,d]
                hold_time = t_windows[seq][1][1]-t_windows[seq][1][0]
                if i_seq==0:
                    d[seq,seq] = dist_p_cir(roi[seq,:],v_maxs[seq],rr[seq],x0)/umax
                if i_seq==len(order)-1:
                    if any(periodic_tasks):
                        for j in np.setdiff1d(periodic_tasks,seq): # periodic tasks except the task itself
                            d[seq,j] = (dist_cir_cir(roi[seq,:],v_maxs[seq],rr[seq],\
                                                roi[j,:],v_maxs[j],rr[j])+hold_time*v_maxs[seq])/umax\
                                                +hold_time
                else:
                    for j in np.union1d(order[i_seq+1],np.setdiff1d(periodic_tasks,seq)): # next target and periodic tasks except the task itself
                        d[seq,j] = (dist_cir_cir(roi[seq,:],v_maxs[seq],rr[seq],\
                                                roi[j,:],v_maxs[j],rr[j])+hold_time*v_maxs[seq])/umax\
                                                +hold_time
            elif types[seq]==4: #G_[a,b]F_[c,d] PERIODIC TASK 
                if i_seq==0:
                    d[seq,seq] = dist_p_cir(roi[seq,:],v_maxs[seq],rr[seq],x0)/umax
                for j in np.setdiff1d(order,seq):
                    d[seq,j] = dist_cir_cir(roi[seq,:],v_maxs[seq],rr[seq],\
                                                roi[j,:],v_maxs[j],rr[j])/umax
            i_seq += 1
        # Start with the first target in the order
        spent_time = d[order[0],order[0]]
        extra_time=[]
        extra_time.append(rr[order[0]]-spent_time)
        # no need to check feasibility at the first tasks!
        if types[order[0]]==4 and spent_time+t_windows[order[0]][1][1]<=hrz:
            flag = 1 # We will update orders here
            period=t_windows[order[0]][1][1]
            visit_e = spent_time
            # Calculate the last visit time acc to realistic remaining times
            temp_sum = 0
            temp_laxity = []
            for j in range(0+1,len(order)): 
                temp_sum=temp_sum+d[order[j-1],order[j]] # These are with real subtask indices
                temp_laxity.append(rr[order[j]]-temp_sum) # j-ind just index\in[0,end]
            visit_l = min(temp_laxity) 
            if visit_e> visit_l:
                #display('failure!:visit_e> visit_l')
                continue
            # BRANCH OUT TO DECOMPOSE
            feasible_orders_new,laxity_vals_new = branch_out(order,0,visit_e,visit_l,d,t_windows,hrz,\
                                             v_maxs,umax,extra_time,r_temp,rr,types)        
            for temp_feas in feasible_orders_new:feasible_orders.append(temp_feas)
            for temp_lax in laxity_vals_new:laxity_vals.append(temp_lax)
            continue
        # continue with the reamining targets
        for i in range(1,len(order)): # Run over the sequence until it is interrupted by a periodic task
            spent_time = spent_time + d[order[i-1],order[i]]
            # Make laxity values vectors - similar to order set!
            extra_time.append(rr[order[i]]-spent_time) # no need to check feasibility at the first tasks!

            if rr[order[i]]-spent_time<0: # check at each step if it is still feasible until there
                flag = 1 # Infeasible
                #display('failure!:rr[order[i]]-spent_time<0')
                break  
            # Periodic task reached which is not at the end and can still be revisited before mission end
            if types[order[i]]==4 and i!=len(order)-1 and spent_time+t_windows[order[i]][1][1]<=hrz:
                flag = 1 # We will update orders here
                period = t_windows[order[i]][1][1]
                visit_e = spent_time
                # Calculate the last visit time acc to realistic remaining times
                temp_sum = 0
                temp_laxity = []
                for j in range(i+1,len(order)): 
                    temp_sum = temp_sum+d[order[j-1],order[j]] # These are with real subtask indices
                    temp_laxity.append(rr[order[j]]-temp_sum) # j-ind just index\in[1,end]
                visit_l = min(temp_laxity) 
                if visit_e> visit_l:
                    break # Infeasible
                    #display('failure!:visit_e> visit_l')
                feasible_orders_new,laxity_vals_new=branch_out(order,i,visit_e,visit_l,d,t_windows,hrz,\
                                                 v_maxs,umax,extra_time,r_temp,rr,types)

                for temp_feas in feasible_orders_new:feasible_orders.append(temp_feas)
                for temp_lax in laxity_vals_new:laxity_vals.append(temp_lax)
                break
        if flag!=1:
            feasible_orders.append(order) # No periodic task is triggered
            laxity_vals.append(extra_time)
    
    temp_sizes = [len(feasible_orders[i]) for i in range(len(feasible_orders))]
    unique_seqs = []
    cost_seqs = []
    portions_seqs = []
    feasible_orders_np = np.array(feasible_orders)

    laxity_vals_np = np.array(laxity_vals)
    try: # If any feasible order is found 
        for i in range(min(temp_sizes),max(temp_sizes)+1): # Sequences with i elements:
            seqs,ic = np.unique(np.vstack(feasible_orders_np[np.where(np.array(temp_sizes)==i)]),axis=0,return_inverse=True)
            unique_seqs.append(seqs)

            seq_total_laxities=np.sum(np.vstack(laxity_vals_np[np.where(np.array(temp_sizes)==i)]),axis=1)
            seq_min_laxities=np.min(np.vstack(laxity_vals_np[np.where(np.array(temp_sizes)==i)]),axis=1)
            seq_laxities = laxity_vals_np[np.where(np.array(temp_sizes)==i)]
            for j in range(int(max(ic))+1): # max(ic) is the number of unique sequences with i elements

                temp_max=np.max(seq_min_laxities[np.where(ic==j)])
                laxity_ind = np.argmax(seq_min_laxities[np.where(ic==j)])
                # CAN CHANGE TO maximum cumulative laxitiy among the same order
                cost_seqs.append(temp_max) 
                temp_laxities = seq_laxities[np.where(ic==j)]
                portions_seqs.append(temp_laxities[laxity_ind])

        #max_laxity=np.max(cost_seqs)
        best_seq_ind = np.argmax(cost_seqs)
        chain = unique_seqs[0][best_seq_ind]
        portions=portions_seqs[best_seq_ind]
        seq_r = r[chain] # Subsequent periodic tasks are dummy.
        min_portions=min(portions)
        return chain,portions,seq_r,min_portions
    except: # No feasible orders
        return [],[],[],[]
    

    
def branch_out(order,ind,visit_e,visit_l,d_out,t_windows,hrz,v_maxs,umax,extra_time,r_out,rr_out,types):
    r = np.copy(r_out)
    rr = np.copy(rr_out)
    d = np.copy(d_out)
    feasible_orders = []
    laxity_vals = []
    extra_time_hit = extra_time[:]
    
    
    for last_hit in [visit_e,visit_l]:
        rr_periodic=[]
        r[order[ind]]=last_hit+t_windows[order[ind]][1][1] # Actually the deadline of the new task to be inserted
        if r[order[ind]]<=hrz:
            spent_time_hit = last_hit # Scalar
            order_new_set = []
            return_time = d[order[ind],order[ind+1]]
            j = ind
            # initialize the dynamic periodic target updates starting by the first+++
            # option - inserting just after the immediate target
            
            if j+1 < len(order)-1:
                rr_periodic.append(min(r[np.hstack([order[ind],order[ind+2:]])])) # Assuming at least two other tasks after the periodic
                d[order[ind+1],order[ind]] = d[order[ind+1],order[ind]]+v_maxs[order[ind]]*rr_periodic[0]/umax # come back
                d[order[ind],order[ind+2]] = d[order[ind],order[ind+2]]+v_maxs[order[ind]]*rr_periodic[0]/umax # keep continue
            else: # Gonna be added to the end
                rr_periodic.append(r[order[ind]])
                d[order[ind+1],order[ind]] = d[order[ind+1],order[ind]]+v_maxs[order[ind]]*rr_periodic[0]/umax # come back
            while return_time+d[order[j+1],order[ind]]<=rr_periodic[j-ind] and j<len(order)-1:
                if j+1<len(order)-1:
                    order_new_set.append(np.hstack([order[0:j+1+1],order[ind],order[j+2:]]))
                    return_time = return_time + d[order[j+1],order[j+2]]
                    j += 1
                    if j+1<len(order)-1:
                        rr_periodic.append(min(r[np.hstack([order[ind],order[j+2:]])]))
                        d[order[j+1],order[ind]] = d[order[j+1],order[ind]]+v_maxs[order[ind]]*rr_periodic[j-ind]/umax # come back
                        d[order[ind],order[j+2]]= d[order[ind+1],order[j+2]]+v_maxs[order[ind]]*rr_periodic[j-ind]/umax # keep continue
                    else: # Gonna be added to the end
                        rr_periodic.append(r[order[ind]])
                        d[order[j+1],order[ind]] = d[order[j+1],order[ind]]+v_maxs[order[ind]]*rr_periodic[j-ind]/umax # come back
                else: # j+1==length(order) % Final LOOP
                    order_new_set.append(np.hstack([order[0:j+1+1],order[ind]]))
                    break

        else:
            order_new_set = [order]
            rr_periodic.append(last_hit)
        k=-1
        for order_new in order_new_set:
            flag=0
            k += 1
            rr[order[ind]] = rr_periodic[k] # Realistic rem time of the trigerring periodic task
            spent_time = spent_time_hit
            extra_time = extra_time_hit[:]

            for i in range(ind+1,len(order_new)): # Run over the sequence until it is interrupted by a periodic task
                spent_time = spent_time + d[order_new[i-1],order_new[i]]
                extra_time.append(rr[order_new[i]]-spent_time) # Make laxity values vectors - similar to order set!

                if rr[order_new[i]]-spent_time<0: # check at each step if it is still feasible until there
                    flag=1                
                    break  
            
                if types[order_new[i]]==4 and i!=len(order_new)-1 and spent_time+t_windows[order_new[i]][1][1]<=hrz: 
                    # Periodic task reached which is not at the end and can still be revisited before mission end
                    flag = 1
                    visit_e = spent_time
                    # Calculate the last visit time acc to realistic remaining times
                    temp_sum=0
                    temp_laxity = []
                    for j in range(i+1,len(order)):
                        temp_sum = temp_sum+d[order[j-1],order[j]] # These are with real subtask indices
                        temp_laxity.append(rr[order[j]]-temp_sum) # j-ind just index\in[1,end]
                    visit_l = min(temp_laxity) #rr(order(i+1))-d(order(i),order(i+1));
                    if visit_e> visit_l:
                        break # Infeasible
                    feasible_orders_new,laxity_vals_new=branch_out(order_new,i,visit_e,visit_l,d,t_windows,hrz,\
                                                     v_maxs,umax,extra_time,r,rr,types)
                    for temp_feas in feasible_orders_new:feasible_orders.append(temp_feas)
                    for temp_lax in laxity_vals_new:laxity_vals.append(temp_lax)
                    break

            if flag!=1:
                feasible_orders.append(order_new) # No periodic task is triggered
                laxity_vals.append(extra_time)
    return feasible_orders,laxity_vals

def interval_clustering(t_windows,periodic_tasks):
    windows = np.copy(t_windows)
    for p in range(len(periodic_tasks)):
        windows[periodic_tasks[p]]=windows[periodic_tasks[p]][1] 
    n_tar = len(windows) # of subformulas
    interval=np.array([], dtype=np.int64).reshape(0,2)
    for i in range(n_tar):
        temp=np.array(windows[i]).flatten()
        interval_new =np.hstack([temp[0]+(len(temp)-2)/2*temp[-2:-1], temp[1]+(len(temp)-2)/2*temp[-1:]])
        interval=np.vstack([interval,interval_new])

    idx=np.argsort(interval[:,1]) # Acc. to end of the interval

    cluster=1000*np.ones(n_tar)

    clst_id=0 # Backward!!!
    cluster[idx[-1:]] = clst_id # Last task
    for i in range(n_tar-1-1,-1,-1):

        if min(interval[np.argwhere(cluster==clst_id),0])>interval[idx[i]][1]:
            clst_id += 1
            cluster[idx[i]] = clst_id # new distinct cluster
        else:
            cluster[idx[i]] = clst_id # belongs to the same cluster
    n_orders = [] # Size of each cluster per element
    order_set_clst = []
    total_orders = 1
    for i in range(int(max(cluster)),-1,-1):
        temp_perms = perms(np.hstack(np.argwhere(cluster==i)))
        order_set_clst.append(temp_perms)
        # total number of possible sequences
        n_orders.append(len(temp_perms)) # Backward again!
        total_orders=total_orders*len(temp_perms)
    if max(cluster)==0:
        completed_orders=order_set_clst
        return completed_orders[0][::-1] 
    order_set_clst=order_set_clst[::-1]


    completed_orders=np.zeros([total_orders,n_tar])
    for i in range(total_orders):
        n = np.unravel_index(i,n_orders)
        temp_order=[]
        for j in range(int(max(cluster)),-1,-1):
            temp_order.append(order_set_clst[j][n[::-1][j]])
        completed_orders[i,:]=np.hstack(temp_order)
    return (completed_orders[::-1].astype(int))

def perms(x):
    """Python equivalent of MATLAB perms."""
    return np.vstack(list(itertools.permutations(x)))[::-1]

#tic = time.time()


# TEST CASES

#if 0:
#    t_windows=[[[0,13],[0,2]],[[0,13],[0,2]]] # STL time windows
#    subformula_types = np.array([3,3]) # 1: F, 2: G, 3: FG, 4: GF | Formula Types
#    x0=np.array([-2,0]) # Initial pos. (x,y) of our system
#    v_tar_max = .1*np.array([.8,.8]) # <- FAST TARGETS NEGATIVE CBF | # SLOW TARGETS (GUANTEE) .1*np.array([.8,.8])
#    umax=.6 # Max vel. of the system
#    roi=.1*np.array([[-5,10,3],[10, 10, 3]])
#    roi_disj = []#np.copy(roi) # Create alternative RoI's (to be modified)  
#    n_tar = np.size(roi,0) # of targets
#    disj_map = np.array([np.arange(0,n_tar)]) 
#    rois = [roi]
#elif 0:
#    t_windows=[[[0,15],[0,15]],[[0,15],[0,15]],[[0,15],[0,15]]] # STL time windows
#    subformula_types = np.array([4,4,4]) # 1: F, 2: G, 3: FG, 4: GF | Formula Types
#    x0=np.array([0,0]) # Initial pos. (x,y) of our system
#    v_tar_max = .1*np.array([.5, .5, .5])
#    umax=.7 # Max vel. of the system
#    roi=.1*np.array([[0,5,3],[4, -3, 3],[-4,-3,3]]); 
#    roi_disj = []#np.copy(roi) # Create alternative RoI's (to be modified)  
#    n_tar = np.size(roi,0) # of targets
#    disj_map = np.array([np.arange(0,n_tar)]) 
#    rois = [roi]
#else:
#    t_windows=[[[0,20]],[[0,20]],[[33,35]],[[0,27],[0,3]],[[0,20],[0,10]]] # STL time windows
#    subformula_types = np.array([1,1,2,3,4]) # 1: F, 2: G, 3: FG, 4: GF | Formula Types
#    x0=np.array([0,0]) # Initial pos. (x,y) of our system
#    v_tar_max = .1*np.array([.6, .6, .4, .5, .55])
#    umax=.7 # Max vel. of the system
#    # % RoI coordinates x1 y1 rad1; x2 y2 rad2 ; ...
#    roi=.1*np.array([[-10, 0, 2],[8.5, 0, 2],[-7.5,-5,2],[-2, -7, 2],[0, 9, 2.5]])
#    roi_disj = np.copy(roi) # Create alternative RoI's (to be modified)
#    n_tar = np.size(roi,0) # of targets
#    alt_inds=np.zeros(n_tar) # Tasks with an alternative will be nonzero only
#    alt_inds[3]=1 # 4th task has alternative in our case     
#    roi_disj[alt_inds>0]=.1*np.array([[-13,-10,2]]) # Alternative target to 4th task
#    disj_map = np.array([np.arange(0,n_tar),np.array([0,0,0,5,0])]) 
#    rois = [roi,roi_disj] # Create alternative full-RoI lists



#chain, rem_time, rem_time_seq, gamma, portions = task_scheduler(rois,t_windows,subformula_types,x0,umax,v_tar_max)
#completed_orders =interval_clustering(t_windows,periodic_tasks)
#print(chain,rem_time,rem_time_seq,gamma)

#elapsed = time.time() - tic
#print({'Elapsed time':elapsed})

