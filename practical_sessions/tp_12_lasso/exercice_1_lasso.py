import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

def main():
    x_values = [1, 10]
    y_values = [1, 10]
    for x_value in x_values:
        for y_value in y_values:
            run_lasso(
                    x_value=x_value,
                    y_value=y_value,
                    )

def run_lasso(x_value, y_value):
    x = np.array([x_value]).reshape(1, -1)
    y = np.array([y_value]).reshape(1, -1)
    lambdas = np.linspace(start=0.1, stop=2*y_value, num=300)
    coefs = list()
    for lambda_ in lambdas:
        estimator = Lasso(alpha=lambda_, fit_intercept=False)
        estimator.fit(X=x, y=y)
        coefs.append(estimator.coef_)

    plt.plot(lambdas, coefs, label=r"$\theta_{\lambda}$")
    title = (
            "Solution of the Lasso regression\n"
            r"$\theta_{\lambda} = \text{argmin}_{\theta}\:\frac{1}{2}(y-\theta)^2+\lambda |\theta| $"
            # f"\nx={x_value}"
            f"\ny={y_value}"
            )
    plt.title(title)
    plt.xlabel(r"$\lambda$")
    plt.legend(loc="best")
    fig_name = f"lasso_regression_x_{x_value}_y_{y_value}.pdf"
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close()


if __name__ == "__main__":
    main()
