"""
Helper functions for emt_tools TODO: Probably have to refactor this file later
"""

import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle, FancyBboxPatch
from sklearn.decomposition import PCA


def pinv(A, rcond=1e-15):
    """
    Compute the pseudo-inverse of a matrix
    """
    U, S, Vh = np.linalg.svd(A, full_matrices=False)
    # Sinv = np.zeros_like(A, dtype=float)
    Sinv = np.diag(np.where(S > rcond, 1.0 / S, 0))

    # print(Vh.shape, S.shape, U.shape)
    Vh = np.matrix(Vh)
    U = np.matrix(U)
    Sinv = np.matrix(Sinv)
    return Vh.H @ Sinv @ U.H


def get_grounded_bases(
    task_dimension,
    hidden_dim,
    W_hh,
    W_hy,
    s,
    alpha=1,
    h_simulated=None,
    f_operator=None,
    strength=1,
    threshold=0.8,
):
    """
    Runs the variable memory computation algorithm grounding the hidden state with input

    :param W_uh: np.ndarray
    :param W_hh: np.ndarray
    :param W_hy: np.ndarray (default: 1)
    :param s: int
        number of steps in the "input phase"
    :param alpha: float in [0, 1]
        alpha controls the amount of grounding between input and outputs.
        alpha = 0: fully grounded in output
        alpha = 1 (default): fully grounded in input
    :param h_simulated: np.ndarray (default: None)
        simulated history of hidden states for computing orthogonal basis. If None, the basis is not computed

    :return:
        Psi: variable memory bases (expanded to full dimensionality)
        Psi_star: dual of the variable memory bases (expanded to full dimensionality)
    """

    # get dimensions

    W_uh = pinv(W_hy)  # ground on W_hy instead of W_uh

    original_s = s

    eig_vals, eig_vecs = np.linalg.eig(W_hh)
    eig_vecs_inv = np.linalg.inv(eig_vecs)

    transient_basis = eig_vecs[:, np.absolute(eig_vals) < threshold]
    transient_inv_basis = eig_vecs_inv[np.absolute(eig_vals) < threshold, :]

    lt_basis = eig_vecs[:, np.absolute(eig_vals) >= threshold]
    lt_inv_basis = eig_vecs_inv[np.absolute(eig_vals) >= threshold, :]

    # print(transient_inv_basis @ transient_basis)
    # plt.show()

    # compute the variable memory basis
    Psi = []
    indices_to_add = []
    for k in range(s, 0, -1):
        if s - k == 0:
            Psi.append(W_uh)
        else:
            Psi.append(np.linalg.matrix_power(W_hh, strength * s - k) @ W_uh)

        Psi[-1] = Psi[-1] / np.linalg.norm(Psi[-1], axis=0).reshape(
            (1, -1)
        )  # normalize the basis

    Psi.reverse()
    Psi = np.concatenate(Psi, axis=1).squeeze()

    ### optimize the basis by reducing the rank of Psi (seems like NNs learn a compressed basis)
    eig_vals, eig_vecs = np.linalg.eig(W_hh)
    print("Matrix size: {}".format(W_hh.shape))
    print("Rank: {}".format(np.sum(np.absolute(eig_vals) > 0.8)))

    ## construct phi
    phi = np.eye(s * task_dimension)
    phi = np.roll(phi, task_dimension)

    phi[:, :task_dimension] = 0
    phi[-task_dimension:, :] = f_operator

    # compute the nullspace of phi
    eig_vals, eig_vecs = np.linalg.eig(phi)

    cur_rank = np.sum(np.absolute(eig_vals) > 0.8)
    ## reduce the rank of \Phi by removing rows and columns
    indices_to_add = []
    psi_projections = []
    for candidate_dim in range(phi.shape[0]):
        psi_projections.append(
            np.linalg.norm(W_hh @ Psi[:, candidate_dim : candidate_dim + 1])
        )
        phi_proposal = np.delete(phi, candidate_dim, 0)
        phi_proposal = np.delete(phi_proposal, candidate_dim, 1)
        eig_vals_proposal, eig_vecs_proposal = np.linalg.eig(phi_proposal)

        new_rank = np.sum(np.absolute(eig_vals_proposal) > 0.8)

        if new_rank < cur_rank:
            indices_to_add.append(candidate_dim)
    ### END optimize basis
    # print(indices_to_add)
    # plt.plot(psi_projections, marker="x", linestyle="None")
    # plt.show()

    Psi = Psi[:, indices_to_add]

    # normalize the basis
    Psi = Psi / np.linalg.norm(Psi, axis=0)

    # compute the dual basis
    Psi_star = pinv(Psi)

    # expanded basis
    Psi_expanded = np.zeros((hidden_dim, original_s * task_dimension)).astype(
        "complex128"
    )
    Psi_star_expanded = np.zeros((original_s * task_dimension, hidden_dim)).astype(
        "complex128"
    )

    Psi_expanded[:, indices_to_add] += Psi
    Psi_star_expanded[indices_to_add, :] += Psi_star

    # print("shapes")
    # print(h_simulated.shape)
    # print(Psi_star.shape)
    # print(Psi.shape)

    h_orthogonal = h_simulated - (Psi @ Psi_star @ h_simulated.T).T
    # print("horth", h_orthogonal.shape)
    # compute the orthogonal basis
    pca = PCA(n_components=10)
    pca.fit(np.array(h_orthogonal))

    Psi_orthogonal = pca.components_.T
    Psi_star_orthogonal = Psi_orthogonal.T

    # print("psi", Psi_orthogonal.shape)

    Psi_expanded = np.zeros(
        (hidden_dim, original_s * task_dimension + pca.components_.shape[0])
    ).astype("complex128")
    Psi_star_expanded = np.zeros(Psi_expanded.T.shape).astype("complex128")

    # add memory basis
    Psi_expanded[:, indices_to_add] += Psi
    Psi_star_expanded[indices_to_add, :] += Psi_star

    # add orthogonal basis
    Psi_expanded[:, original_s * task_dimension :] += Psi_orthogonal
    Psi_star_expanded[original_s * task_dimension :, :] += Psi_star_orthogonal

    return Psi_expanded, Psi_star_expanded


def plot_evolution_in_bases(
    u_history,
    h_history,
    y_history,
    Psi_star,
    Phi,
    task_dimension,
    s,
    filename="animation.gif",
):
    """
    Plot the evolution of the RNN states in the basis for which Psi_star is the dual
    """
    neuron_radius = 0.1

    vb_circle_radius = 2 * neuron_radius * task_dimension

    hstate_x = []
    hstate_y = []
    inputstate_x = [-vb_circle_radius - 2.5 * neuron_radius - 1.5] * task_dimension
    inputstate_y = [
        -vb_circle_radius - 1 - i * (2 * neuron_radius + neuron_radius // 4)
        for i in range(task_dimension)
    ]

    outputstate_x = [vb_circle_radius + 2.5 * neuron_radius + 1.5] * task_dimension
    outputstate_y = [
        -vb_circle_radius - 1 - i * (2 * neuron_radius + neuron_radius // 4)
        for i in range(task_dimension)
    ]

    radii = [neuron_radius] * (s * task_dimension)

    # print(h_history.shape, Psi_star.shape)

    h_basis = h_history @ Psi_star.T

    delta_theta = 2 * np.pi / s
    cur_theta = (s - 1) * delta_theta - np.pi / 2
    for i in range(s):
        x_cur = vb_circle_radius * np.cos(cur_theta)
        y_cur = (
            vb_circle_radius * np.sin(cur_theta) + (task_dimension // 2) * neuron_radius
        )

        for j in range(task_dimension):
            hstate_x.append(x_cur)
            hstate_y.append(y_cur)

            y_cur -= 2 * neuron_radius + neuron_radius // 4

        cur_theta -= delta_theta

    # Build plot
    fig, ax = plt.subplots(figsize=(6, 4))
    hstate_patches = []
    inputstate_patches = []
    outputstate_patches = []

    for x1, y1, r in zip(hstate_x, hstate_y, radii):
        circle = Circle((x1, y1), r)
        hstate_patches.append(circle)

    for x1, y1, r in zip(inputstate_x, inputstate_y, radii[:task_dimension]):
        circle = Circle((x1, y1), r)
        inputstate_patches.append(circle)

    for x1, y1, r in zip(outputstate_x, outputstate_y, radii[:task_dimension]):
        circle = Circle((x1, y1), r)
        outputstate_patches.append(circle)

    ## add hiddenstate rectangle
    ax.add_patch(
        FancyBboxPatch(
            (
                -vb_circle_radius - 0.5,
                -vb_circle_radius - task_dimension * neuron_radius - 0.5,
            ),
            2 * (vb_circle_radius + 0.5),
            2 * vb_circle_radius + task_dimension * neuron_radius + 1.5,
            color="black",
            alpha=0.1,
        )
    )

    # add these circles to a collection
    p_hstate = PatchCollection(hstate_patches, cmap="coolwarm", alpha=1.0)
    ax.add_collection(p_hstate)

    p_inputstate = PatchCollection(inputstate_patches, cmap="coolwarm", alpha=1.0)
    ax.add_collection(p_inputstate)

    p_outputstate = PatchCollection(outputstate_patches, cmap="coolwarm", alpha=1.0)
    ax.add_collection(p_outputstate)

    ax.set_xlim(-vb_circle_radius - 0.5, vb_circle_radius + 0.5)
    ax.set_ylim(
        -vb_circle_radius - (task_dimension // 2) * neuron_radius - 3.0,
        vb_circle_radius + (task_dimension // 2) * neuron_radius + 1.5,
    )

    ax.axis("equal")
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_title("Step {} ({} phase)".format(1, "input"), fontweight="bold")

    plt.tight_layout()

    def update(num):
        p_hstate.set_array(
            Psi_star @ h_history[num, :].flatten()
        )  # set new color colors
        p_inputstate.set_array(u_history[num].flatten())
        p_outputstate.set_array(y_history[num].flatten())

        # Scale plot ax
        phase = "input"
        if num >= s:
            phase = "output"
        ax.set_title("Step {} ({} phase)".format(num + 1, phase), fontweight="bold")

        p_hstate.set_clim(-0.1, 0.1)
        p_inputstate.set_clim(-0.3, 0.3)
        p_outputstate.set_clim(-0.3, 0.3)

        return p_hstate, p_inputstate, p_outputstate

    ani = matplotlib.animation.FuncAnimation(
        fig, update, frames=h_history.shape[0], interval=1000, repeat=True
    )
    ani.save(filename, writer="imagemagick", fps=1)

    pass


def spectral_comparison(operator1, operator2, threshold=0.9):
    """
    Compare the spectral properties of two linear operators.
    The eigenvalues less than threshold is fully removed and the spectrum is compared.

    R = \{ (\lamda^1_i, \lambda^2_i) \forall i \in max(|\lambda^1|, |\lambda^2|) \}
    argmin_{R} \sum_{|Arg(\lambda^1_i) - Arg(\lambda^2_i)|}

    :param operator1: np.ndarray (phi)
    :param operator2: np.ndarray (w_hh)
    :param threshold: float (default 1-1e-3)
    :return:
        avg_error: float
            If the operators have the same rank and same eigenvalue arguments avg_error->0
        -1: int
            If the operators have different ranks
    """
    eigenvalues1, eigenvectors1 = np.linalg.eig(operator1)
    eigenvalues2, eigenvectors2 = np.linalg.eig(operator2)

    # remove eigenvalues less than threshold
    evals1_reduced = eigenvalues1[np.absolute(eigenvalues1) > threshold]
    evals2_reduced = eigenvalues2[np.absolute(eigenvalues2) > threshold]

    print(evals1_reduced.shape, evals2_reduced.shape)
    ## make sure evals2 has atleast evals1 number of evals
    if evals1_reduced.shape[0] > evals2_reduced.shape[0]:
        evals2_reduced = np.concatenate(
            (
                evals2_reduced,
                np.zeros(evals1_reduced.shape[0] - evals2_reduced.shape[0]),
            )
        )
    elif evals1_reduced.shape[0] < evals2_reduced.shape[0]:
        evals1_reduced = np.concatenate(
            (
                evals1_reduced,
                np.zeros(evals2_reduced.shape[0] - evals1_reduced.shape[0]),
            )
        )

    n = evals1_reduced.shape[0]
    n2 = evals2_reduced.shape[0]

    # compute the spectral distance
    evals1_vec = np.zeros((n, 2))
    evals2_vec = np.zeros((n2, 2))

    evals1_vec[:, 0] = evals1_reduced.real
    evals1_vec[:, 1] = evals1_reduced.imag
    evals1_vec /= np.linalg.norm(evals1_vec, axis=1).reshape((-1, 1))

    evals2_vec[:, 0] = evals2_reduced.real
    evals2_vec[:, 1] = evals2_reduced.imag
    evals2_vec /= np.linalg.norm(evals2_vec, axis=1).reshape((-1, 1))

    err = 0
    # compute the complex argument error
    for i in range(n):
        delta = np.arccos(
            (evals1_vec[i : i + 1].reshape((1, -1)) @ evals2_vec.T).flatten()
        )
        delta_argmin = np.argmin(np.absolute(delta))

        err += np.absolute(delta[delta_argmin])

        # remove the used eigenvalues
        evals2_vec = np.delete(evals2_vec, delta_argmin, axis=0)

    return err / n


def construct_phi_from_operator(f_operator):
    output_dim, input_dim = f_operator.shape
    seq_length = output_dim // input_dim

    ## construct phi
    phi = np.eye(seq_length * input_dim)
    phi = np.roll(phi, input_dim)

    phi[:, :input_dim] = 0
    phi[-input_dim:, :] = f_operator.T

    eig_vals, eig_vecs = np.linalg.eig(phi)

    ## The below section compresses the rank of phi by removing irrelevant rows and columns
    cur_nullspace_rank = np.sum(np.absolute(eig_vals) > 0.8)
    indices_to_add = []
    for candidate_dim in range(phi.shape[0]):
        phi_proposal = np.delete(phi, candidate_dim, 0)
        phi_proposal = np.delete(phi_proposal, candidate_dim, 1)
        eig_vals_proposal, eig_vecs_proposal = np.linalg.eig(phi_proposal)

        new_nullspace_rank = np.sum(np.absolute(eig_vals_proposal) > 0.8)

        if new_nullspace_rank < cur_nullspace_rank:
            indices_to_add.append(candidate_dim)

    basis = np.eye(phi.shape[0])

    for dim in range(basis.shape[0]):
        if dim not in indices_to_add:
            basis[dim, :] = 0
            basis[:, dim] = 0

    return basis @ phi @ basis.T
