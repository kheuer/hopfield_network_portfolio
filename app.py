import base64
from flask import Flask, render_template, request, redirect, url_for, session, send_file
import matplotlib.pyplot as plt
import os
from network import HopfieldNetwork
import logging
from io import BytesIO
import time
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Required for session management

networks = {}
n_neurons_default = 25**2


def drop_outdated_networks():
    for key, _ in networks.items():
        time_saved = float(key.split("_")[0])
        print(key, time_saved)
        if time.time() - time_saved >= 60 * 60 * 24:
            networks.pop(key)


def get_network():
    """Get the HopfieldNetwork instance from the session."""
    return networks[session["identifier"]]


def init_session():
    if not "identifier" in session:
        session["identifier"] = f"{time.time()}_{request.remote_addr}"
    if not session["identifier"] in networks:
        networks[session["identifier"]] = HopfieldNetwork(n_neurons_default)


@app.route("/hopfield/clear")
def clear():
    if "identifier" in session:
        if session["identifier"] in networks:
            networks.pop(session["identifier"])
        session.pop("identifier")
    return redirect(url_for("index"))


@app.route("/", methods=["POST", "GET"])
def index():
    init_session()

    if request.method == "POST":
        if "n_neurons" in request.form:
            print("reset")
            reset_network()

        elif "ch_selection" in request.form:
            save_pattern()

        elif "n_steps" in request.form:
            advance()

        elif "solve" in request.form:
            solve()

        elif "randomize" in request.form:
            randomize()

        elif "mutate" in request.form:
            mutate()

        elif "set_state_to_pattern" in request.form:
            set_to_pattern()

    network = get_network()
    pattern_image = visualize_state(network, network.state)

    if network.patterns is None:
        saved_patterns = []
    else:
        saved_patterns = []
        for i in range(network.patterns.shape[1]):
            pattern = network.patterns[:, i]
            saved_patterns.append(visualize_state(network, pattern))

    return render_template(
        "index.html",
        pattern_image=pattern_image,
        saved_patterns=saved_patterns,
        n_neurons=network.n_neurons,
        is_stable=network.is_in_local_minima(),
    )


def reset_network():
    n_neurons = int(request.form["n_neurons"])
    networks[session["identifier"]] = HopfieldNetwork(n_neurons)


def save_pattern():
    network = get_network()
    pattern_number = request.form.get("ch_selection")
    state = network.get_number_pattern(int(pattern_number))
    network.save_pattern(state)
    network.train()


def advance():
    network = get_network()
    steps = int(request.form["n_steps"])
    network.run(steps)
    return redirect(url_for("index"))


def solve():
    network = get_network()
    network.solve()


def randomize():
    drop_outdated_networks()  # drop outdated here because this is sometimes used but not always

    network = get_network()
    network.set_random_pattern()


def mutate():
    network = get_network()
    network.set_mutated_pattern()


def visualize_state(network, state):
    """Generate and save the visualization of the current state in memory."""
    fig = network.visualize(state)

    # Save the plot to a BytesIO object
    img_bytes = BytesIO()
    fig.savefig(img_bytes, format="png")
    img_bytes.seek(0)

    plt.close(fig)
    return base64.b64encode(img_bytes.getvalue()).decode("utf-8")


def set_to_pattern():
    pattern_id = int(request.form.get("set_state_to_pattern"))

    network = get_network()
    if network.patterns is None:
        return

    network.state = np.copy(network.patterns[:, pattern_id])


if __name__ == "__main__":
    app.run(debug=False)
