import math
import torch
import gpytorch

class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ZeroMean(), num_tasks=2
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()) + gpytorch.kernels.ConstantKernel(),
            num_tasks=2, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x, validate_args=True, interleaved=True)
    

def gpytorch_test(num_iter:int):
    train_x1=50*torch.randint(1, 11, (100,)) # so-far observed ub
    train_x2=torch.stack([torch.randint(5, int(ub.item()), (1,)) for ub in train_x1]) # so-far observed gn
    train_y1=50*torch.randint(1, 11, (100,)) # so-far estimated ub
    train_y2=torch.stack([torch.randint(5, int(ub.item()), (1,)) for ub in train_y1]) # so-far estimated gn
    train_x=torch.stack([train_x1, train_x2], dim=-1)
    train_y=torch.stack([train_y1, train_y2], dim=-1)

    N_task=2
    likelihood=gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=N_task)
    model=MultitaskGPModel(train_x, train_y, likelihood)


    
    optimizer=torch.optim.Adam(model.parameters(), lr=0.01)

    mll=gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(num_iter):
        
        model.train()
        optimizer.zero_grad()
        output=model(train_x)
        loss=-mll(output, train_y)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f' % (i + 1, num_iter, loss.item()))
        optimizer.step()        

        model.eval()
        likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_x=torch.linspace(0, 1, 51)
            N_t=test_x.shape[0]
            posterior=model(test_x)
            
            covFull=posterior.covariance_matrix # (N_t * num_tasks, N_t * num_tasks)

            jointUncertainty=torch.zeros(test_x.shape)
            for i in range(N_t):
                startRow=i*N_task
                endRow=startRow+N_task
                covTask=covFull[startRow:endRow, startRow:endRow]
                ju=torch.log2(torch.linalg.det(covTask)*(2*math.pi*math.exp(1))**2)
                jointUncertainty[i]=ju
            
            minIdx=torch.argmin(jointUncertainty)


            

