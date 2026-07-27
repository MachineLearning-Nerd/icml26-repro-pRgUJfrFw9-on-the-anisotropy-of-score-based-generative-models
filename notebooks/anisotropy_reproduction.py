import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    return mo, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    # Paper-scale SAD endpoint reproduction

    The paper asks whether an untrained score network's architecture creates
    preferred output directions. It defines **Score Anisotropy Directions
    (SADs)** as eigenvectors of the average output-geometry matrix and reports
    that an iDDPM generates rank-one data better along small-eigenvalue SADs.

    This notebook opens with the completed evidence. It embeds the signed
    five-pair result, so viewing it does not rerun training or generation.
    """)
    return


@app.cell
def _(np):
    sw2 = np.array(
        [2.523301275184064, 2.69050719535384, 6.424866323309012,
         3.2828236837755913, 4.516650849364211]
    )
    msw2 = np.array(
        [19.183560007281415, 13.691540406317294, 24.375159079083495,
         16.52711137478938, 20.75802328491255]
    )
    intervals = {
        "SW2": (2.7420881249702798, 5.279937038942132),
        "MSW2": (15.828264140870534, 22.082122379390484),
    }
    return intervals, msw2, sw2


@app.cell
def _(intervals, msw2, np, plt, sw2):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, values, name, color in zip(
        axes, [sw2, msw2], ["SW2", "MSW2"], ["#16697A", "#C04A2B"]
    ):
        mean = float(values.mean())
        lo, hi = intervals[name]
        ax.bar(np.arange(5), values, color=color, alpha=0.85)
        ax.axhspan(lo, hi, color="#F2C14E", alpha=0.28, label="exact 95% interval")
        ax.axhline(mean, color="#7B2CBF", linestyle="--", linewidth=2,
                   label=f"mean = {mean:.3f}")
        ax.axhline(0, color="#222222", linewidth=1)
        ax.set(title=name, xlabel="paired seed",
               ylabel="largest SAD − smallest SAD")
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("All five paper-scale endpoint pairs favor the smallest SAD")
    fig
    return


@app.cell
def _(intervals, mo, msw2, sw2):
    mo.md(
        f"""
        ## What was measured

        Lower Sliced Wasserstein distance is better, so every positive bar is in
        the paper-consistent direction.

        | Metric | Mean largest − smallest | Exact paired-bootstrap 95% interval |
        | --- | ---: | ---: |
        | SW2 | `{sw2.mean():.6f}` | `[{intervals["SW2"][0]:.6f}, {intervals["SW2"][1]:.6f}]` |
        | MSW2 | `{msw2.mean():.6f}` | `[{intervals["MSW2"][0]:.6f}, {intervals["MSW2"][1]:.6f}]` |

        The verifier enumerates all `5^5 = 3,125` ordered paired resamples.
        Reversing largest and smallest labels makes both effects negative and is
        rejected by the same acceptance rule.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Why this is still BLOCKED

    The geometry was estimated from one million fresh iDDPM networks, split
    into two independent 500,000-network shards. The leading eigenvector is
    stable across shards (absolute overlap `0.997695`), but the smallest
    eigenvector has overlap only `0.199398`. All seven audited interior
    directions also fail the preregistered individual-vector gate.

    The experiment therefore gives faithful **endpoint corroboration**, not a
    verification of the full monotonic rank claim. Running the interior sweep
    would assign causal meaning to arbitrary rotations inside unstable
    eigenspaces.

    The fixed formal command is:

    ```text
    uv sync --locked && .venv/bin/python repro/run_campaign.py
    ```

    The accepted aggregate is run
    `0c6c3210-941e-443c-ab16-dd524c804ac9` at Git
    `26b31ae7f0bb19c7901c72f54413326ecd729d9e`, CPU only.
    """)
    return


if __name__ == "__main__":
    app.run()
