import sys
sys.path.append('C:/Users/Shahar/IML.HUJI')
from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"
from utils import *

def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    samples = np.random.normal(10, 1, 1000)
    estimator = UnivariateGaussian()
    estimator.fit(samples)
    bins = 100

    fig = make_subplots(rows=5, cols=1, specs=[[{"rowspan": 4, "secondary_y": True}], [None], [None], [None], [{}]])\
        .add_trace(go.Histogram(x=samples, opacity=0.75, bingroup=1, nbinsx=bins), secondary_y=False)\
        .add_trace(go.Scatter(x=samples, y=[0]*samples.shape[0], mode='markers', opacity=0.75), row=5, col=1)

    fig.update_layout(title_text="$\\text{Histograms of }1000\\text{ Samples from }X\\sim\\mathcal{N}\\left(10,1\\right)$")\
        .update_yaxes(title_text="Number of samples", secondary_y=False, row=1, col=1)\
        .update_yaxes(showgrid=False, row=5, col=1, showticklabels=False)\
        .update_xaxes(showgrid=False, title_text="Value", row=5, col=1)\
        .update_xaxes(showticklabels=False, row=1, col=1)\
        .update_layout(showlegend=False)

    fig.show()
    print(estimator.mu_, estimator.var_)

    # Question 2 - Empirically showing sample mean is consistent
    ms = np.linspace(10, 1000, 100).astype(np.int)
    mu, sigma = 10, 1
    estimated_mean_dist, estimated_var_dist = [], []
    for m in ms:
        estimator.fit(samples[0:m])
        estimated_mean_dist.append(np.abs(estimator.mu_ - mu)) 
        estimated_var_dist.append(np.abs(estimator.var_ - sigma ** 2)) 

    go.Figure([go.Scatter(x=ms, y=estimated_mean_dist, mode='markers+lines', name=r'$|\hat\mu - \mu|$'),
               go.Scatter(x=ms, y=estimated_var_dist, mode='markers+lines', name=r'$|\hat var - var|$')],
            layout=go.Layout(title=r"$\text{The absolute distance between the estimated and true value of the expectation as a function of the sample size}$", 
                    xaxis_title="$\\text{number of samples}$", 
                    yaxis_title="absolute distance between the estimated and true value",
                    height=300)).show()

    # Question 3 - Plotting Empirical PDF of fitted model
    estimator.fit(samples)
    pdfs = np.array(estimator.pdf(samples))
    go.Figure(go.Scatter(x=samples, y=pdfs, mode='markers'),
            layout=go.Layout(title=r"$\text{Empirical PDF of fitted model}$", 
                    xaxis_title="$\\text{sample value}$", 
                    yaxis_title="pdf value",
                    height=300)).show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    raise NotImplementedError()

    # Question 5 - Likelihood evaluation
    raise NotImplementedError()

    # Question 6 - Maximum likelihood
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    # test_multivariate_gaussian()
