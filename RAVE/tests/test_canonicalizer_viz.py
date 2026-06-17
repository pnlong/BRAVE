import numpy as np

from rave.fader.canonicalizer_viz import plot_latent_domain_scatter


def test_plot_latent_domain_scatter_pca():
    rng = np.random.default_rng(0)
    in_domain = rng.standard_normal((80, 128)).astype(np.float32)
    ood = rng.standard_normal((60, 128)).astype(np.float32) + 2.0
    fig = plot_latent_domain_scatter(in_domain, ood, method="pca")
    assert fig.axes
    import matplotlib.pyplot as plt
    plt.close(fig)
