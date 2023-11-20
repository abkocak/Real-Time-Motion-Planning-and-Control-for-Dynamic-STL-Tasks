# Real-Time-Motion-Planning-and-Control-for-Dynamic-STL-Tasks
The code for regenerating the scenarios in the paper "Sequential Control Barrier Functions for Mobile Robots with Dynamic Temporal Logic Specifications."

In this paper, we address a motion planning and control problem for mobile  robots to satisfy rich and time-varying tasks expressed as Signal Temporal Logic specifications. The specifications may include tasks with nested temporal operators or time-conflicting requirements (e.g., achieving periodic tasks or tasks defined within the same time interval). Moreover, the tasks can be defined in locations changing with time (i.e., dynamic targets), and their future motions are not known a priori. This unpredictability requires an online control approach which motivates us to investigate the use of control barrier functions (CBFs). The proposed CBFs take into account the actuation limits of the robots and a feasible sequence of STL tasks. They define time-varying feasible sets of states the system must always stay inside. We show the feasible sequence generation process that even includes the decomposition of periodic tasks and alternative scenarios due to disjunction operators. The sequence is used to define CBFs, ensuring STL satisfaction. We also show some theoretical results on the correctness of the proposed method. We illustrate the benefits of the proposed method and analyze its performance via simulations and experiments with aerial robots.

## Video
Please check the experiments [video](https://youtu.be/whg_X1dy_es). It highlights some key points in the paper.

https://github.com/abkocak/Real-Time-Motion-Planning-and-Control-for-Dynamic-STL-Tasks/assets/80661909/7d08dbca-eac9-4a5f-ab2f-e351aec6b524



#### The code features:
<ul type="square">
<!-- li><code>todo</code> </li -->
    <li>Feasible scheduling of STL tasks in "task_scheduler.py" module;</li>
    <li>Case 1: The main scenario with a complex STL specification defined over dynamic tasks moving on predefined trajectories with maximum velocity;</li>
    <li>Case 2: The same scenario with randomly moving targets;</li>
    <li>Case 3: Another specification comprised of only periodic tasks;</li>
    <li>Case 4: A specification defined over adversarial targets that are repelled by the controlled system;</li>
    <li>Case 5: The same adversarial scenario with much faster targets;</li>
</ul>

#### Requirements:
<ul type="square">
<!-- li><code>todo</code> </li -->
    <li>CasADi — an open-source tool for nonlinear optimization and algorithmic differentiation in <a href="https://web.casadi.org" target="_blank">https://web.casadi.org</a>;</li>
    <li>Crazyswarm — for running the code with actual crazyflie drones (optional) in <a href="https://github.com/USC-ACTLab/crazyswarm" target="_blank">https://github.com/USC-ACTLab/crazyswarm</a>;</li>
</ul>

#### Paper:

Ali Tevfik Buyukkocak, Derya Aksaray, and Yasin Yazicioglu. "Sequential Control Barrier Functions for Mobile Robots with Dynamic Temporal Logic Specifications," 2023.

