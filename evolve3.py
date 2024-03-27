# Necessary libraries and modules are imported here
import os
import sys
import logging
import argparse
import yaml
import re
from pathlib import Path
import joblib
import optuna
from optuna.multi_objective import trial as optuna_trial

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov8') not in sys.path:
    sys.path.append(str(ROOT / 'yolov8'))  # add yolov5 ROOT to PATH
if str(ROOT / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'strong_sort'))  # add strong_sort ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from yolov8.ultralytics.yolo.utils import LOGGER
from yolov8.ultralytics.yolo.utils.checks import check_requirements, print_args
from track import run

class Objective(Evaluator):
    """Objective function to evolve the best set of hyperparameters."""
    
    def __init__(self, opts):  
        self.opt = opts
                
    def get_new_config(self, trial):
        """Overwrites the tracking config by newly generated hyperparameters"""
        
        d = {}
        self.opt.conf_thres = trial.suggest_float("conf_thres", 0.35, 0.55)
        
        if self.opt.tracking_method == 'strongsort':
            # Provide suggestions for hyperparameters appropriate for the strongsort algorithm
            pass
        elif self.opt.tracking_method == 'ocsort':
            # Provide suggestions for hyperparameters appropriate for the ocsort algorithm
            pass
        # Include other code for other algorithms here
        
    def __call__(self, trial: optuna_trial.Trial):
        """Objective function for multi-objective optimization."""
        
        # Provide suggestions for hyperparameters and evaluate the results
        pass


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', type=str, default=WEIGHTS / 'yolov8m.pt', help='model.pt path(s)')
    parser.add_argument('--reid-weights', type=str, default=WEIGHTS / 'osnet_x1_0_dukemtmcreid.pt')
    parser.add_argument('--tracking-method', type=str, default='strongsort', help='strongsort, ocsort')
    parser.add_argument('--tracking-config', type=Path, default=None)
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--project', default=ROOT / 'runs' / 'evolve', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--benchmark', type=str,  default='MOT17', help='MOT16, MOT17, MOT20')
    parser.add_argument('--split', type=str,  default='train', help='existing project/name ok, do not increment')
    parser.add_argument('--eval-existing', type=str, default='', help='evaluate existing tracker results under mot_callenge/MOTXX-YY/...')
    parser.add_argument('--conf-thres', type=float, default=0.45, help='confidence threshold')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[1280], help='inference size h,w')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--n-trials', type=int, default=10, help='nr of trials for evolution')
    parser.add_argument('--resume', action='store_true', help='resume hparam search')
    parser.add_argument('--processes-per-device', type=int, default=2, help='how many subprocesses can be invoked per GPU (to manage memory consumption)')
    parser.add_argument('--objectives', type=str, default='HOTA,MOTA,IDF1', help='set of objective metrics: HOTA,MOTA,IDF1')
    
    opt = parser.parse_args()
    opt.tracking_config = ROOT / 'trackers' / opt.tracking_method / 'configs' / (opt.tracking_method + '.yaml')
    opt.objectives = opt.objectives.split(",")

    device = []
    
    for a in opt.device.split(','):
        try:
            a = int(a)
        except ValueError:
            pass
        device.append(a)
    opt.device = device
        
    print_args(vars(opt))
    return opt


class ContinuousStudySave:
    """Helper class for saving the study after each trial. This is to avoid
       losing partial study results if the study is stopped before finishing"""

    def __init__(self, tracking_method):
        self.tracking_method = tracking_method
        
    def __call__(self, study, trial):
        joblib.dump(study, opt.tracking_method + "_study.pkl")

    
if __name__ == "__main__":
    opt = parse_opt()
    check_requirements(('optuna', 'plotly', 'kaleido', 'joblib', 'pycocotools'))
    
    if opt.resume:
        # Resume optimization from the last saved study
        study = joblib.load(opt.tracking_method + "_study.pkl")
    else:
        # Create a new study if not resuming
        study = optuna.create_study(directions=['maximize']*len(opt.objectives))
        # Configure the first trial with params from yaml file
        with open(opt.tracking_config, 'r') as f:
            params = yaml.load(f, Loader=yaml.loader.SafeLoader)
            study.enqueue_trial(params[opt.tracking_config.stem])

    continuous_study_save_cb = ContinuousStudySave(opt.tracking_method)
    study.optimize(Objective(opt), n_trials=opt.n_trials, callbacks=[continuous_study_save_cb])
        
    # Write the best parameters to the config file for the selected tracking method
    write_best_HOTA_params_to_config(opt, study)
    
    # Save the hps study; all trial results are stored here, used for resuming
    joblib.dump(study, opt.tracking_method + "_study.pkl")
    
    # Generate plots
    save_plots(opt, study, opt.objectives)
    # Print the best trial metric results
    print_best_trial_metric_results(study, opt.objectives)
