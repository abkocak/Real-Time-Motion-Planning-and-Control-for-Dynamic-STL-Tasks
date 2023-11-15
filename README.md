# Real-Time-Motion-Planning-and-Control-for-Dynamic-STL-Tasks
The code for regenerating the scenarios in the paper "Sequential Control Barrier Functions under Signal Temporal Logic Specifications with Dynamic and Uncooperative Targets."

In this paper, We address a motion planning and control problem for a family of dynamical systems to satisfy rich and timevarying tasks expressed as Signal Temporal Logic specifications. The specifications may include tasks with nested temporal operators or time-conflicting requirements (e.g., achieving periodic tasks or tasks defined within the same time interval). Moreover, the tasks can be defined in target locations changing with time (i.e., dynamic targets), and their future motions are not known a priori. This unpredictability requires an online control approach which motivates us to investigate the use of control barrier functions (CBFs). The proposed CBFs take into account the actuation limits of the dynamical system and a feasible sequence of STL tasks. They define time-varying feasible sets of states the system must always stay inside. We show the feasible sequence generation process that even includes the decomposition of periodic tasks and alternative scenarios due to disjunction operators. The sequence is used to define CBFs ensuring the STL satisfaction.We also show some theoretical results on the correctness of the proposed method. We illustrate the benefits of the proposed method and analyze its performance via simulations and experiments.

## Video
Please check the experiments [video](https://youtu.be/s7T0bHtP5qE). It highlights some key points in the paper.

https://github.com/abkocak/Real-Time-Motion-Planning-and-Control-for-Dynamic-STL-Tasks/assets/80661909/30f31c51-ff33-49ae-afdb-52da7b4bfd42


#### The code features:
<ul type="square">
<!-- li><code>todo</code> </li -->
    <li>Feasible scheduling of STL tasks in "task_scheduler.py" module;</li>
    <li>Case 1: The main scenario with a complex STL specification defined over dynamic tasks moving on predefined trajectories with maximum velocity;</li>
    <li>Case 2: The same scenario with randomly moving targets;</li>
    <li>Case 3: Another specification comprised of only periodic tasks;</li>
    <li>Case 4: A specification defined over adversarial targets which are repelled by the controlled system;</li>
    <li>Case 5: The same adversarial scenario with much faster targets;</li>
</ul>

#### Requirements:
<ul type="square">
<!-- li><code>todo</code> </li -->
    <li>CasADi — an open-source tool for nonlinear optimization and algorithmic differentiation in <a href="https://web.casadi.org" target="_blank">https://web.casadi.org</a>;</li>
    <li>Crazyswarm — for running the code with actual crazyflie drones (optional) in <a href="https://github.com/USC-ACTLab/crazyswarm" target="_blank">https://github.com/USC-ACTLab/crazyswarm</a>;</li>
</ul>

#### Paper:

Ali Tevfik Buyukkocak, Derya Aksaray, and Yasin Yazicioglu. "Control of Mobile Robots with Sequential Barrier Functions under Temporal Logic Specifications on Noncooperative Targets." Robotics and Autonomous Systems (RAS), 2023 (In Review).

