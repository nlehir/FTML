import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def norm_pca_basis(sample, component_1, component_2) -> float:
    projection_1 = np.dot(sample, component_1)
    projection_2 = np.dot(sample, component_2)
    norm = np.sqrt(projection_1**2 + projection_2**2)
    return norm


def main() -> None:
    data = np.load("Exercices_5_data.npy")

    """
    Plot the data
    """
    plt.scatter(x=data[:, 0], y=data[:, 1], alpha=0.7)
    plt.title("Raw data")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig("1_raw_data.pdf")
    plt.close()

    """
    Plot the centered data
    """
    data = data - data.mean(axis=0)
    plt.scatter(x=data[:, 0], y=data[:, 1], alpha=0.7)
    plt.title("Centered data")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig("2_centered_data.pdf")

    """
    Apply PCA
    """
    pca = PCA()
    pca.fit(data)
    explained_variance_1 = pca.explained_variance_ratio_[0]
    explained_variance_2 = pca.explained_variance_ratio_[1]

    """
    Plot the first two components
    """
    component_1 = pca.components_[0]
    component_2 = pca.components_[1]
    x_component_1, y_component_1 = component_1[0], component_1[1]
    x_component_2, y_component_2 = component_2[0], component_2[1]
    label_1 = f"Component 1: variance ratio {explained_variance_1:.2f}"
    label_2 = f"Component 2: variance ratio {explained_variance_2:.2f}"
    plt.arrow(
        x=0, y=0, dx=x_component_1, dy=y_component_1, label=label_1, color="orange"
    )
    plt.arrow(
        x=0, y=0, dx=x_component_2, dy=y_component_2, label=label_2, color="yellow"
    )
    plt.legend(loc="best")
    plt.title("PCA components")
    plt.savefig("3_data_with_pca_components.pdf")

    """
    Force equal scales on x and y
    """
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    plt.title("PCA components (same x and y scale)")
    plt.savefig("4_data_with_pca_components_ratio_1.pdf")
    plt.close()

    """
    Check the consistency of the norm comutation,
    up to numerical rounding errors
    """
    for sample in data:
        sample_norm_canonical = np.linalg.norm(sample)
        sample_norm_pca_basis = norm_pca_basis(
            sample=sample, component_1=component_1, component_2=component_2
        )
        equal = np.isclose(a=sample_norm_canonical, b=sample_norm_pca_basis)
        print(f"\n{sample_norm_canonical=}")
        print(f"{sample_norm_pca_basis=}")
        print(f"{equal=}")


if __name__ == "__main__":
    main()
