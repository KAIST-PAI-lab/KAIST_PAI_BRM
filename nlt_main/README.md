# Python Implementation of Number-line task

### Requirements
This code requires Python 3.10 to run.
Please make sure to install all required packages listed in ```requirements.txt``` by running the following command:

```
pip install -r requirements.txt
```

### How to execute?
Run the following command to start the task:
```
python main.py
```

Currently, the experiment consists of 5 practice trials to help participants become familiar with the task, followed by two main blocks of 45 trials each, resulting in a total of 90 experimental trials. 

Throughout the experiment, perceptual cues such as the dot size and cumulative area are controlled, as suggested in [paper](https://link.springer.com/article/10.3758/s13423-021-02041-5). On half of the trials, the individual dot sizes were identical across boxes, making the cumulative area scale with the number of dots. On the other half, the cumulative area was kept constant, resulting in smaller dot sizes as the number increased. 

As GPAL package has not been fully implemented, the upper-bound and given-numbers were randomly selected instead.