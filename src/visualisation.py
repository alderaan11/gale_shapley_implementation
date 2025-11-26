import matplotlib.pyplot as plt
import numpy as np

import json
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import typer
import seaborn as sns


def plot_satisfaction_vs_scorefinal(results):
    avg_satisf = [(r["topk_global"] + r["satisfaction_croisee"]) / 2 for r in results]
    scoref = [r["score_final"] for r in results]
    labels = [res.get("label", f"M{i}") for i, res in enumerate(results)]

    plt.figure()
    for x, y, label in zip(avg_satisf, scoref, labels):
        plt.scatter([x], [y])
        plt.text(x, y, label)

    plt.xlabel("Average Satisfaction (Top-K + Satisfaction Croisée)")
    plt.ylabel("Score Final")
    plt.title("Satisfaction vs Score Final")
    plt.grid(True)
    plt.show()




def plot_score_vs_weights(
    score_fn,
    matching,
    matching_ideal_etu,
    matching_ideal_uni,
    prefs_etus,
    prefs_unis,
    k=3,
    steps=21
):

    alphas = np.linspace(0, 1, steps)
    betas  = np.linspace(0, 1, steps)

    A, B, Scores = [], [], []

    for a in alphas:
        for b in betas:
            g = 1 - a - b
            if g < 0:
                continue

            res = score_fn(
                matching,
                matching_ideal_etu,
                matching_ideal_uni,
                prefs_etus,
                prefs_unis,
                k,
                a, b, g
            )

            A.append(a)
            B.append(b)
            Scores.append(res["score_final"])

    A = np.array(A)
    B = np.array(B)
    Scores = np.array(Scores)

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(A, B, c=Scores, cmap="viridis")
    plt.colorbar(scatter, label="Score final")

    plt.xlabel("alpha (poids Top-K)")
    plt.ylabel("beta (poids Satisfaction croisée)")
    plt.title("Variation du Score Final\n(gamma = 1 - alpha - beta)")
    plt.grid(True)
    plt.show()


def plot_matching_components(results):
    labels = [r["label"] for r in results]

    # Extraire les scores
    topk_vals = [r["topk_global"] for r in results]
    cross_vals = [r["satisfaction_croisee"] for r in results]
    optimal_vals = [r["optimalité"] for r in results]
    final_vals = [r["score_final"] for r in results]

    x = np.arange(len(labels))
    width = 0.20

    plt.figure(figsize=(12, 6))

    plt.bar(x - 1.5*width, topk_vals, width, label="Top-K")
    plt.bar(x - 0.5*width, cross_vals, width, label="Satisfaction croisée")
    plt.bar(x + 0.5*width, optimal_vals, width, label="Optimalité (1 - regret)")
    plt.bar(x + 1.5*width, final_vals, width, label="Score final")

    plt.xticks(x, labels)
    plt.ylim(0, 1)
    plt.ylabel("Valeur")
    plt.title("Comparaison des composantes entre matchings")
    plt.legend()
    plt.grid(True, axis="y", linestyle="--", alpha=0.6)

    plt.show()


def plot_score_vs_weights_recompute(
    score_fn,
    matching,
    matching_ideal_etu,
    matching_ideal_uni,
    prefs_etus,
    prefs_unis,
    k=3,
    step=0.1
):
    """
    Recalcule le score final pour toutes les valeurs de alpha, beta
    avec un step donné (par exemple 0.1).
    gamma = 1 - alpha - beta
    """
    from score import score_final
    alphas = np.arange(0, 1 + step, step)
    betas  = np.arange(0, 1 + step, step)

    A, B, Scores = [], [], []

    for a in alphas:
        for b in betas:
            g = 1 - a - b
            if g < 0:
                continue  # gamma ne peut pas être négatif

            res = score_final(
                matching_ideal_etu,
                matching_ideal_uni,
                prefs_etus,
                prefs_unis,
                k,
                a, b, g
            )

            A.append(a)
            B.append(b)
            Scores.append(res["score_final"])

    # Convertir en numpy
    A = np.array(A)
    B = np.array(B)
    Scores = np.array(Scores)

    # --- Heatmap Scatter ---
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(A, B, c=Scores, cmap="viridis", s=200, edgecolors='black')

    plt.colorbar(scatter, label="Score final")
    plt.xlabel("alpha (poids Top-K)")
    plt.ylabel("beta (poids Satisfaction croisée)")
    plt.title(f"Variation du Score Final\n(step = {step}, gamma = 1 - alpha - beta)")
    plt.grid(True)
    plt.show()



app = typer.Typer()

@app.command()
def main():

    import json
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    from pathlib import Path

    sns.set(style="darkgrid", context="talk")  

    result_path = Path("./results_swing")
    records = []


    for time_dir in result_path.iterdir():
        json_file = time_dir / "preference_et_resultats.json"
        if not json_file.exists():
            continue

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for matching_name, result in data["scores"].items():
            details = result["details"]

            records.append({
                "run_dir": time_dir.name,
                "matching": matching_name,
                "score_final": result["score_final"],
                "alpha": details["alpha"],
                "beta": details["beta"],
                "gamma": details["gamma"],
                "k": details["k"]
            })

    df = pd.DataFrame(records)
    

    print("\n==============================")
    print(" MOYENNES DES SCORES PAR PARAMÈTRE ")
    print("==============================")

    print("\n-> Moyenne du score final par alpha :")
    print(df.groupby("alpha")["score_final"].mean())

    print("\n-> Moyenne du score final par beta :")
    print(df.groupby("beta")["score_final"].mean())

    print("\n-> Moyenne du score final par gamma :")
    print(df.groupby("gamma")["score_final"].mean())

    print("\n-> Moyenne du score final par k :")
    print(df.groupby("k")["score_final"].mean())
    print("\n==============================")
    print(" MOYENNES PAR COMBINAISON (alpha, beta, gamma, k) ")
    print("==============================")

    df_mean_full = df.groupby(["alpha", "beta", "gamma", "k"])["score_final"].mean()
    print(df_mean_full)

    print("\n\n\n\n")
    df_mean = df.groupby(["alpha", "beta", "gamma", "k"]).agg(
    score_moyen=("score_final", "mean"),
    count=("score_final", "count")
    ).reset_index()

    print("\n==============================")
    print(" MOYENNES PAR COMBINAISON UNIQUE (alpha, beta, gamma, k) ")
    print("==============================\n")
    print(df_mean.to_string(index=False))

    parameters = ["alpha", "beta", "gamma", "k"]
    colors = sns.color_palette("tab10", len(parameters))

    plt.figure(figsize=(12, 7))

    for color, param in zip(colors, parameters):
        plt.scatter(df[param], df["score_final"], s=80, alpha=0.75,
                    label=param.upper(), color=color)

    plt.xlabel("Paramètres")
    plt.ylabel("Score final")
    plt.title("Score final en fonction de chaque paramètre (visualisation améliorée)")
    plt.legend()
    plt.tight_layout()
    plt.show()



    sns.pairplot(df, vars=["alpha", "beta", "gamma", "k"],
        hue="score_final", palette="viridis")
    plt.show()




    plt.figure(figsize=(12, 7))

    plt.scatter(df["alpha"], df["score_final"], s=80, c=df["k"], cmap="viridis")
    plt.xlabel("alpha")
    plt.ylabel("Score final")
    plt.title("Score final en fonction de alpha et k (sans annotations)")
    plt.colorbar(label="k")
    plt.tight_layout()
    plt.show()



    df_mean = df.groupby(["alpha", "beta", "gamma", "k"]).agg(
        score_moyen=("score_final", "mean"),
        count=("score_final", "count")
    ).reset_index()

    plt.figure(figsize=(14, 8))

    plt.scatter(
        df_mean["alpha"], df_mean["score_moyen"],
        s=180, c=df_mean["k"], cmap="magma",
        marker="o", alpha=0.85,
        label="alpha"
    )

    plt.scatter(
        df_mean["beta"], df_mean["score_moyen"],
        s=180, c=df_mean["k"], cmap="magma",
        marker="s", alpha=0.85,
        label="beta"
    )


    plt.scatter(
        df_mean["gamma"], df_mean["score_moyen"],
        s=180, c=df_mean["k"], cmap="magma",
        marker="^", alpha=0.85,
        label="gamma"
    )

    plt.xlabel("Valeur du paramètre")
    plt.ylabel("Score moyen")
    plt.title("Score moyen selon alpha (o), beta (□) et gamma (△)")
    plt.colorbar(label="k")
    plt.legend()
    plt.tight_layout()
    plt.show()



    df_mean = df.groupby(["alpha", "beta", "gamma", "k"]).agg(
        score_moyen=("score_final", "mean"),
        count=("score_final", "count")
    ).reset_index()

    plt.figure(figsize=(12, 7))
    plt.scatter(df_mean["alpha"], df_mean["score_moyen"],
                s=200, c=df_mean["k"], cmap="magma", alpha=0.8)
    plt.xlabel("alpha")
    plt.ylabel("Score moyen")
    plt.title("Score moyen par combinaison (sans annotations)")
    plt.colorbar(label="k")
    plt.tight_layout()
    plt.show()

    df_mean = df.groupby(["alpha", "beta", "gamma", "k"]).agg(
    score_moyen=("score_final", "mean"),
    count=("score_final", "count")
    ).reset_index()

    plt.figure(figsize=(12, 7))

    plt.scatter(
        df_mean["alpha"],
        df_mean["score_moyen"],
        s=200,
        c=df_mean["k"],
        cmap="magma",
        alpha=0.8
    )

    plt.xlabel("alpha")
    plt.ylabel("Score moyen")
    plt.title("Score moyen par combinaison")
    plt.colorbar(label="k")

    # --- 🔍 Ajouter l’annotation (alpha, beta, gamma, k) ---
    for _, row in df_mean.iterrows():
        label = f"a={row.alpha:.2f}\nb={row.beta:.2f}\ng={row.gamma:.2f}\nk={int(row.k)}"
        plt.annotate(
            label,
            (row.alpha, row.score_moyen),
            textcoords="offset points",
            xytext=(5, 5),     # Décalage pour lire le texte
            ha="left",
            fontsize=8
        )

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))

    params = ["alpha", "beta", "gamma", "k"]

    means = [
        df_mean.groupby(p)["score_moyen"].mean().mean()
        for p in params
    ]

    plt.bar(params, means, color=["#ff6f61", "#6b5b95", "#88b04b", "#f7cac9"])

    plt.ylabel("Score final moyen")
    plt.title("Influence moyenne des paramètres sur le score final")
    plt.grid(axis="y", alpha=0.3)
    plt.show()

    df_corr = df_mean[["alpha", "beta", "gamma", "k", "score_moyen"]]

    plt.figure(figsize=(8, 6))
    sns.heatmap(df_corr.corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Corrélation entre paramètres et score final")
    plt.show()

if __name__ == "__main__":
    app()