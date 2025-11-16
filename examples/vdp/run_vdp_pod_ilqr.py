import numpy as np
import sys
import os
from pathlib import Path


# import Iterative_LQR
# from Iterative_LQR.src.main_ilqr import iLQR

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from sim_vdp import SimulateVDP
from vdp_params import *
from main_ilqr import iLQR
from main_pod_ilqr import POD_iLQR
from ltv_sys_id import LTV_SysID
from arma_ltv_sys_id import ARMA_LTV_SysID

class RunVdp(SimulateVDP):

    def __init__(self, mu, state_dimension, control_dimension, dt, init_state, C):
        SimulateVDP.__init__(self, mu, state_dimension, control_dimension, dt, init_state)
        self.C = C

    def simulate_step(self,x,u):
        return (self.C @ super().simulate_trajectory(x, u)[-1])
    
    def simulate_trajectory(self, y_init=0, u = np.array([0.0]), horizon=1):
        return (self.C @ super().simulate_trajectory(y_init, u, horizon).T).T


if __name__=="__main__":

    cwd = os.getcwd()
    path_to_vdp = Path(cwd)/"examples/vdp"
    MODEL = path_to_vdp/"models/None.xml"

    path_to_export = path_to_vdp/"VDP_Experiments/exp_1"
    path_to_policy_file = path_to_export/"vdp_policy.txt"
    path_to_cost_file = path_to_export / "training_cost_data.txt"
    path_to_training_cost_fig = path_to_export/"episodic_cost_training.png"
    path_to_traj_fig = path_to_export/"optimal_traj.png"
    # path_to_data = path_to_export / "vdp_D2C_data.txt"

    init_state = np.zeros((state_dimension,1))
    init_state[0] = .2
    print('Initial phase : \n', init_state)

    # Create model instance
    run_vdp = RunVdp(mu, state_dimension, control_dimension, dt, init_state, C)

    obs_init_state = C @ init_state
    final_state = np.zeros((n_aug,1))
    Q = Q_aug
    Q_final = Q_final_aug
    print('Goal phase : \n', final_state)
	
    # No. of ILQR iterations to run
    n_iterations = 10

    # u_opt = np.array([-1.215601127007042, -0.06800620455962883, 0.23526238769837085, 0.29748412447401523, 0.28919180547032036, 0.26585666280432274, 0.24170351612588648, 0.21496601789473885, 0.18938218320160433, 0.1684568909941789, 0.14663722971617615, 0.1275977127135976, 0.11080624805365258, 0.09163208235824732, 0.07685863342110148, 0.06571546592484064, 0.058686111625737586, 0.07357592999448705, 0.15207540645619244, 0.46257158223018763]).reshape(horizon,control_dimension,1)
    # u_opt = np.array([-1.215601127007042, -0.06800619906307068, 0.23526236975837933, 0.2974840965531259, 0.28919166237989635, 0.26585619803513, 0.24170177676943214, 0.21495982734344224,     0.18935984938999395, 0.16837699696543895, 0.14635200579609758, 0.12658509214158573, 0.1072332892612731, 0.07914183244796774, 0.03378058642770091, -0.07888349273508148, -0.39802143322876854, -1.1003431578753535, -0.9859738183703414, -3.4611024559834296]).reshape(horizon,control_dimension,1)
    # u_opt = np.array([0.04967141530112327, -0.013826427592062644, 0.06476901253479994, 0.15230285828600137, -0.023414737420843202, -0.023412833237604805, 0.15792595043387636, 0.07675866435213399,     -0.04689085743213717, 0.054456377842058076, -0.045623840735976036, -0.04402720137391134, 0.033182936338050036, -0.15991487158550327, -0.06412962567347066, 0.3075227381476168, 1.0478883774420915, 2.9850452072308564, 2.802482955890577, 9.684507833098467]).reshape(horizon,control_dimension,1)

    # Create iLQR instance
    ilqr = POD_iLQR(run_vdp, state_dimension, control_dimension, alpha, horizon, obs_init_state, final_state, n_z, q, q_u, Q, Q_final, R, 
                nominal_init_stddev, n_sys_id_samples=100, pert_sys_id_sigma=1e-5, arma_sys_id_flag = True)
    ilqr.iterate_ilqr(n_iterations)

    ilqr.plot_episodic_cost_history(path_to_training_cost_fig)
    ilqr.save_policy(path_to_policy_file)
    ilqr.save_cost(path_to_cost_file)

    # Check and Simulate the obtained policy
    run_vdp.simulate_trajectory(y_init = init_state.flatten(), u = ilqr.U.flatten(), horizon=horizon)
    run_vdp.draw_figure(path_to_traj_fig)

    # Test sys_id
    # arma_ltv_sysid = ARMA_LTV_SysID(run_vdp, state_dimension, control_dimension, n_z, C, q, q_u, horizon, n_samples=500, pert_sigma = 1e-3)
    # x_t = np.array([2.0,0.0]).reshape(state_dimension,1)
    # u_t = np.array([0]).reshape(control_dimension,1)
    
    # A, B = arma_ltv_sysid.traj_sys_id((np.concatenate((ilqr.X_0.reshape(1, ilqr.n_x, 1), ilqr.X), axis=0)), ilqr.U.reshape(horizon, control_dimension))
    # print(A, B)