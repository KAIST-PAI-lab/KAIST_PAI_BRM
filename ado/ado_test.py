from adopy import Task, Model, Engine
import numpy as np
from scipy.stats import norm
import time

N_MAX = 500

def mixed_log_linear(a, b, lam, given_number, N_MAX):
    return a*((1-lam)*given_number + lam*N_MAX/np.log(N_MAX)*np.log(given_number))+b

def generate_grid_params():
    return {
        'a': np.arange(0.5, 1.1, 0.1),
        'b': np.arange(0, 101, 10),
        'lam': np.arange(0.1, 1.1, 0.1),
        'sigma': np.arange(1, 11, 1)
    }

def generate_grid_designs():
    return {'given_number': np.arange(5, 501, 1)}

def generate_grid_response():
    return {'response': np.arange(5, 501, 5)}

def log_likelihood(response, given_number, a, b, lam, sigma):
    x = given_number
    y = response

    # 예측값
    pred = a * ((1 - lam) * x + lam * N_MAX / np.log(N_MAX) * np.log(x)) + b

    # 정규분포 기반 log-likelihood
    return norm(loc=pred, scale=sigma).logpdf(y)

if __name__ == "__main__":
    task = Task(name='ADO-NLT', designs=['given_number'], responses=['response'])
    model = Model(name = "MLLM",
              task = task,
              params = ["a", "b", "lam", "sigma"],
              func = log_likelihood)
    print("Initializing ADO-NLT...")
    start_time = time.time()
    print(f"time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    engine = Engine(task, model, 
                    grid_design=generate_grid_designs(),
                    grid_param=generate_grid_params(),
                    grid_response=generate_grid_response()
    )
    print("ADO-NLT initialized successfully.")
    print(f"time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    print(f"Initialization took {time.time() - start_time:.2f} seconds.")
    # engine.reset()

    results = []
    for i in range(10):
        design = engine.get_design('optimal')
        print(f"Design {i+1}: {design}")
        response = int(input("Enter response estimate: "))

        results.append((design['given_number'], response))

        engine.update(design, response)

    print("Final Results:")
    for given_number, response in results:
        print(f"Given Number: {given_number}, Response: {response}")