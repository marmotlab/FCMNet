# FCMNet: Full Communication Memory Net for Team-Level Cooperation in Multi-Agent Systems
This is the code for implementing the FCMNet algorithm presented in the paper which will appear AAMAS 2022:"FCMNet: Full Communication Memory Net for Team-Level Cooperation in Multi-Agent Systems"

## Requirements
```
pip install -r requirements.txt
```

## Running Code
First modify parameters in ```alg_parameters.py``` and ```env_parameters.py```, then call ```python main.py```

## Key Files
```FCMNet.py``` - Defines network architecture

```alg_parameters.py``` - Algorithm parameters

```env_parameters.py``` - Environment parameters

```env_wrapper.py``` - Build a multiprocessing environment and communicates with subproceses via pipes

```evaluation.py``` - Evaluate the trained model using a greedy strategy

```learner.py``` - Update model and record performance

```main.py``` - Driver of program

```model.py``` - Build tensor graph and calculate flow

```policy.py``` - Build critic and actor

```runner.py``` - Run multiple episodes in the multiprocessing environment

## Other Links
install starcraft environment - https://github.com/oxwhirl/smac
